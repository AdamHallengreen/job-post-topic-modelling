# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic-ai-slim[anthropic]>=2.66.0",
#     "polars[calamine]>=1.32.2",
# ]
# ///
"""Code Jobnet job posts for working hours, flexibility and job traits with Claude.

prompt.txt is sent as the (cached) system prompt and each post in data/Jobnet.xlsx (Title + Text)
as the user message. The API forces the reply to match the JobPostCoding schema below: 16 items
with a quote and a 0/1 flag, plus a validity flag. Results are appended to a JSONL checkpoint as
they arrive, so an interrupted run resumes where it stopped, and are exported to a wide parquet
file at the end.

Run from the repo root (real runs need ANTHROPIC_API_KEY):
    uv run scripts/llm_topics/annotate_jobnet.py --dry-run   # show prompt + schema, no API call
    uv run scripts/llm_topics/annotate_jobnet.py --n 50      # pilot on a random sample
    uv run scripts/llm_topics/annotate_jobnet.py             # all posts, live requests
    uv run scripts/llm_topics/annotate_jobnet.py --batch     # all posts, Batch API (50% cheaper, slower)
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

import polars as pl
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.capabilities import BatchModelRequests
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicEffort, AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DATA_PATH = ROOT / "data" / "Jobnet.xlsx"
PROMPT_PATH = SCRIPT_DIR / "prompt.txt"
DEFAULT_OUT_DIR = ROOT / "data" / "llm_topics"
DEFAULT_MODEL = "claude-haiku-5-5"
MAX_TOKENS = 4000  # reply is ~300 tokens; the rest is room for thinking
FATAL_STATUS = {401, 403, 404}  # bad key, no access, unknown model: stop instead of failing every post


# --- Output schema ------------------------------------------------------------------------------
# Field names match prompt.txt. Fields are generated in the order written, so `evidence` comes
# before `present`: the model has to find a quote before it decides.


class Item(BaseModel):
    evidence: str
    present: Literal[0, 1]


class IrregularHours(BaseModel):
    Evening: Item
    Night: Item
    Weekends: Item
    ShiftWork: Item


class Flexibility(BaseModel):
    wfhOptions: Item
    flexibilityHoursCompany: Item
    flexibilityHoursEmployee: Item
    flexibleTasks: Item


class Goldin(BaseModel):
    timePressure: Item
    contactWithOthers: Item
    interpersonalRelationships: Item
    structuredWork: Item
    freedomToMakeDecisions: Item


class OtherTraits(BaseModel):
    Administrative: Item
    AI: Item


class Validity(BaseModel):
    reason: str
    is_job_post: Literal[0, 1]


class JobPostCoding(BaseModel):
    IrregularHours: IrregularHours
    flexibility: Flexibility
    GoldinTraits: Goldin
    OtherTraits: OtherTraits
    valid: Validity


# --- Inputs -------------------------------------------------------------------------------------


def load_posts(path: Path) -> pl.DataFrame:
    """Load posts with a 0-based `row` key (sheet order); ID is not unique, so it is only carried along."""
    cols = ["ID", "Title", "Text"]
    df = (
        pl.read_excel(path, columns=cols, schema_overrides=dict.fromkeys(cols, pl.String))
        .with_row_index("row")
        .with_columns(pl.col("row").cast(pl.Int64), pl.col("Title", "Text").fill_null(""))
    )
    empty = pl.col("Text").str.strip_chars() == ""
    print(f"Loaded {df.height:,} posts; skipping {df.filter(empty).height} with empty Text")
    return df.filter(~empty)


def user_prompt(post: dict) -> str:
    body = "\n\n".join(part for part in (post["Title"].strip(), post["Text"].strip()) if part)
    return f"<job_post>\n{body}\n</job_post>"


def make_agent(
    instructions: str, model: str, effort: AnthropicEffort, concurrency: int, batch: bool
) -> Agent[None, JobPostCoding]:
    client = AsyncAnthropic(max_retries=8)  # retries 429/5xx with backoff
    return Agent(
        AnthropicModel(model, provider=AnthropicProvider(anthropic_client=client)),
        instructions=instructions,  # system prompt, identical for every post
        output_type=NativeOutput(JobPostCoding),  # the API enforces this JSON schema
        model_settings=AnthropicModelSettings(
            max_tokens=MAX_TOKENS,
            anthropic_effort=effort,
            anthropic_cache_instructions=True,  # system prompt is read from cache after the first post
        ),
        # In batch mode every pending run must be able to queue its request, so no concurrency cap.
        max_concurrency=None if batch else concurrency,
        capabilities=[BatchModelRequests()] if batch else None,
    )


# --- Checkpoint files ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_rows(raw_path: Path, prompt_sha: str) -> set[int]:
    records = read_jsonl(raw_path)
    if any(r["prompt_sha256"] != prompt_sha for r in records):
        sys.exit(f"{raw_path} was made with a different prompt.txt. Use another --out-dir or move the file.")
    return {r["row"] for r in records}


def export_parquet(raw_path: Path, parquet_path: Path) -> None:
    rows = []
    for rec in {r["row"]: r for r in read_jsonl(raw_path)}.values():  # one record per row
        out = {"row": rec["row"], "ID": rec["ID"], "Title": rec["Title"]}
        output = rec["output"]
        for group, items in output.items():
            if group == "valid":
                continue
            for name, item in items.items():
                out[name] = item["present"]
                out[f"{name}_evidence"] = item["evidence"]
        out["valid"] = output["valid"]["is_job_post"]
        out["valid_reason"] = output["valid"]["reason"]
        for key in ("model", "effort", "prompt_sha256", "input_tokens", "cache_read_tokens", "output_tokens"):
            out[key] = rec.get(key)
        rows.append(out)
    if not rows:
        print(f"No results in {raw_path} yet; nothing to export.")
        return
    pl.from_dicts(rows, infer_schema_length=None).sort("row").write_parquet(parquet_path)
    print(f"Wrote {len(rows):,} coded posts to {parquet_path}")


# --- Running ------------------------------------------------------------------------------------


async def code_post(agent: Agent[None, JobPostCoding], post: dict) -> tuple[dict, dict | Exception]:
    """Code one post. Returns the post and either the result fields or the exception."""
    try:
        result = await agent.run(user_prompt(post))
    except Exception as exc:
        return post, exc
    usage = result.usage
    return post, {
        "output": result.output.model_dump(),
        "input_tokens": usage.input_tokens,  # includes cache_read_tokens
        "cache_read_tokens": usage.cache_read_tokens,
        "output_tokens": usage.output_tokens,
    }


async def run_posts(agent: Agent[None, JobPostCoding], todo: list[dict], meta: dict, out: dict) -> None:
    tasks = [asyncio.create_task(code_post(agent, post)) for post in todo]
    n_ok = n_err = 0
    totals = {"input_tokens": 0, "cache_read_tokens": 0, "output_tokens": 0}
    with out["raw"].open("a", encoding="utf-8") as raw, out["errors"].open("a", encoding="utf-8") as errors:
        for i, done in enumerate(asyncio.as_completed(tasks), start=1):
            post, result = await done
            record = {"row": post["row"], "ID": post["ID"], "Title": post["Title"], **meta}
            record["time"] = datetime.now(UTC).isoformat(timespec="seconds")
            if isinstance(result, Exception):
                n_err += 1
                errors.write(json.dumps({**record, "error": repr(result)}, ensure_ascii=False) + "\n")
                errors.flush()
                if isinstance(result, ModelHTTPError) and result.status_code in FATAL_STATUS:
                    sys.exit(f"Stopping: {result!r}")
            else:
                n_ok += 1
                for key in totals:
                    totals[key] += result[key]
                raw.write(json.dumps({**record, **result}, ensure_ascii=False) + "\n")
                raw.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i:,}/{len(todo):,} done ({n_err} errors)")
    if n_ok:
        means = ", ".join(f"{key} {value / n_ok:,.0f}" for key, value in totals.items())
        print(f"Mean tokens per post: {means}")
    if n_err:
        print(f"{n_err} posts failed (see {out['errors']}); re-run the same command to retry them.")


def dry_run(instructions: str, posts: pl.DataFrame) -> None:
    """Print what would be sent for the first post, without calling the API."""
    post = posts.row(0, named=True)
    print("=== System prompt (prompt.txt) ===")
    print(instructions)
    print(f"\n=== User message (row {post['row']}, ID {post['ID']}) ===")
    print(user_prompt(post))
    print("\n=== Output schema enforced by the API ===")
    print(json.dumps(JobPostCoding.model_json_schema(), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Anthropic model id (default: %(default)s)")
    parser.add_argument(
        "--effort", default="low", choices=get_args(AnthropicEffort), help="Thinking effort (default: %(default)s)"
    )
    parser.add_argument("--n", type=int, help="Code a random sample of N posts instead of all of them")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the --n sample (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel live requests (default: %(default)s)")
    parser.add_argument("--batch", action="store_true", help="Use the Batch API (50%% cheaper, up to 24h)")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output folder (default: data/llm_topics)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the prompt for one post; no API call")
    parser.add_argument("--export-only", action="store_true", help="Only rebuild the parquet from saved results")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    args = parse_args()
    tag = f"{args.model}_{args.effort}"
    out = {
        "raw": args.out_dir / f"raw_{tag}.jsonl",
        "errors": args.out_dir / f"errors_{tag}.jsonl",
        "parquet": args.out_dir / f"jobnet_llm_{tag}.parquet",
    }
    if args.export_only:
        export_parquet(out["raw"], out["parquet"])
        return

    instructions = PROMPT_PATH.read_text(encoding="utf-8").strip()
    prompt_sha = hashlib.sha256(instructions.encode("utf-8")).hexdigest()

    posts = load_posts(DATA_PATH)
    if args.n is not None:
        posts = posts.sample(n=min(args.n, posts.height), seed=args.seed, shuffle=True)
    if args.dry_run:
        dry_run(instructions, posts)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = load_done_rows(out["raw"], prompt_sha)
    todo = posts.filter(~pl.col("row").is_in(list(done))).to_dicts()
    print(f"{args.model} (effort {args.effort}): {posts.height - len(todo):,} already coded, {len(todo):,} to go")
    if todo:
        agent = make_agent(instructions, args.model, args.effort, args.concurrency, args.batch)
        meta = {"model": args.model, "effort": args.effort, "prompt_sha256": prompt_sha}
        asyncio.run(run_posts(agent, todo, meta, out))
    export_parquet(out["raw"], out["parquet"])


if __name__ == "__main__":
    main()
