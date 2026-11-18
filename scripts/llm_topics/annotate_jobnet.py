# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "claude-agent-sdk",
#     "pydantic-ai-slim[anthropic]>=2.66.0",
#     "polars[calamine]>=1.32.2",
# ]
# ///
"""Code Jobnet job posts with Claude.

For each post in data/Jobnet.xlsx, Claude reads the instructions in prompt.txt and the post, and
answers in the JobPostCoding format below: for each of 15 items, a quote from the post and a 0/1.

    uv run scripts/llm_topics/annotate_jobnet.py --n 50   # pilot: a random sample of 50 posts
    uv run scripts/llm_topics/annotate_jobnet.py          # all posts
    add --api to pay per token with ANTHROPIC_API_KEY instead of using your Claude subscription
    add --model claude-sonnet-5-5 to use another model than MODEL below
    add --effort high to use another effort level than EFFORT below

The DVC stage llm_pilot runs this script with --metrics, which also writes summary numbers for the
sampled posts to a JSON file, so `dvc exp show` can compare pilots with different models.

Every answer is saved as soon as it arrives, so running the same command again continues where
the last run stopped. See README.md in this folder for the full guide.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import polars as pl
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.anthropic import AnthropicModelSettings

# --- Settings: edit these to try another effort level ------------------------------------------

MODEL = "claude-haiku-5-5"  # used unless you pass --model
Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT: Effort = "low"  # how much Claude thinks first; used unless you pass --effort
POSTS_AT_A_TIME = 3  # posts sent in parallel; keep this low on a subscription

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DATA_PATH = ROOT / "data" / "Jobnet.xlsx"
SYSTEM_PROMPT = (SCRIPT_DIR / "prompt.txt").read_text(encoding="utf-8").strip()
PROMPT_ID = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8]


def output_folder(model: str, effort: Effort) -> Path:
    """One folder per model, effort and version of prompt.txt, so different setups never mix.

    It holds results.jsonl (one line per coded post), errors.jsonl (one line per failed attempt)
    and results.parquet (one wide row per coded post, rebuilt after every run).
    """
    return ROOT / "data" / "llm_topics" / f"{model}_{effort}_prompt-{PROMPT_ID}"


# --- The answer format --------------------------------------------------------------------------
# Field names match prompt.txt. Claude writes the fields in this order, so it has to find a quote
# (`evidence`) before it decides (`present`).


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


class Goldin(BaseModel):
    timePressure: Item
    contactWithOthers: Item
    interpersonalRelationships: Item
    structuredWork: Item
    freedomToMakeDecisions: Item


class OtherTraits(BaseModel):
    Administrative: Item
    AI: Item
    partTime: Item


class Validity(BaseModel):
    reason: str
    is_job_post: Literal[0, 1]


class JobPostCoding(BaseModel):
    IrregularHours: IrregularHours
    flexibility: Flexibility
    GoldinTraits: Goldin
    OtherTraits: OtherTraits
    valid: Validity


# --- Posts --------------------------------------------------------------------------------------


def load_posts() -> pl.DataFrame:
    """All posts with text, plus a `row` number (0 = first post in the sheet), since ID isn't unique."""
    cols = ["ID", "Title", "Text"]
    posts = pl.read_excel(DATA_PATH, columns=cols, schema_overrides=dict.fromkeys(cols, pl.String))
    posts = posts.with_row_index("row").with_columns(pl.col("row").cast(pl.Int64), pl.col(cols).fill_null(""))
    return posts.filter(pl.col("Text").str.strip_chars() != "")


def post_text(post: dict) -> str:
    return f"{post['Title'].strip()}\n\n{post['Text'].strip()}".strip()


def user_message(post: dict) -> str:
    return f"<job_post>\n{post_text(post)}\n</job_post>"


# --- Option 1 (default): your Claude subscription, through Claude Code -------------------------
# The Claude Agent SDK runs Claude Code, logged in with your Pro/Max account, once per post.
# The options switch off everything Claude Code normally loads (tools, your plugins and hooks,
# claude.ai connectors, CLAUDE.md, memory), so Claude sees only prompt.txt and the post.


def subscription_options(model: str, effort: Effort) -> ClaudeAgentOptions:
    os.environ.pop("ANTHROPIC_API_KEY", None)  # if set, a key would be used (and billed) instead
    return ClaudeAgentOptions(
        model=model,
        effort=effort,
        system_prompt=SYSTEM_PROMPT,  # replaces Claude Code's own system prompt
        output_format={"type": "json_schema", "schema": JobPostCoding.model_json_schema()},
        tools=[],
        setting_sources=[],
        env={
            "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
        },
        extra_args={"no-session-persistence": None, "strict-mcp-config": None, "disable-slash-commands": None},
    )


async def code_with_subscription(options: ClaudeAgentOptions, post: dict) -> dict:
    answer = None
    async for message in query(prompt=user_message(post), options=options):
        if isinstance(message, ResultMessage):  # the final message: the answer and token counts
            answer = message
    if answer is None or answer.structured_output is None:
        raise RuntimeError(answer.result if answer else "Claude Code ended without an answer")
    usage = answer.usage or {}
    cached = usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
    return {
        "output": JobPostCoding.model_validate(answer.structured_output).model_dump(),
        "input_tokens": usage.get("input_tokens", 0) + cached,
        "output_tokens": usage.get("output_tokens", 0),
    }


# --- Option 2 (--api): the Anthropic API, through PydanticAI -----------------------------------
# A PydanticAI Agent bundles three things: which model to call, the instructions (system prompt)
# and the type the answer must have. `await agent.run(text)` sends one request and returns the
# answer as a JobPostCoding object in `result.output`. Needs ANTHROPIC_API_KEY.


def api_agent(model: str, effort: Effort) -> Agent[None, JobPostCoding]:
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # hide PydanticAI's welcome banner
    return Agent(
        f"anthropic:{model}",
        instructions=SYSTEM_PROMPT,
        output_type=NativeOutput(JobPostCoding),  # the API itself makes the answer match the format
        model_settings=AnthropicModelSettings(
            anthropic_effort=effort,
            anthropic_cache_instructions=True,  # after the first post, the prompt costs ~10%
        ),
    )


async def code_with_api(agent: Agent[None, JobPostCoding], post: dict) -> dict:
    result = await agent.run(user_message(post))
    return {
        "output": result.output.model_dump(),
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
    }


# --- Running ------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def code_posts(posts: list[dict], code, client, folder: Path) -> None:
    """Code the posts a few at a time and save each answer (or error) in `folder` right away."""
    n_failed = 0
    for start in range(0, len(posts), POSTS_AT_A_TIME):
        group = posts[start : start + POSTS_AT_A_TIME]
        # Send the group in parallel. return_exceptions=True: a failed post gives its error
        # instead of stopping the others.
        answers = await asyncio.gather(*(code(client, post) for post in group), return_exceptions=True)
        for post, answer in zip(group, answers, strict=True):
            record = {"row": post["row"], "time": datetime.now(UTC).isoformat(timespec="seconds")}
            if isinstance(answer, BaseException):
                n_failed += 1
                append_jsonl(folder / "errors.jsonl", {**record, "error": repr(answer)})
            else:
                append_jsonl(folder / "results.jsonl", {**record, **answer})
        print(f"  {start + len(group):,}/{len(posts):,} posts, {n_failed} failed", end="\r")
        if all(isinstance(answer, BaseException) for answer in answers):
            # Usually the usage limit, a login problem or a wrong model name, which no retry fixes.
            print(f"\nStopping: every post in the last group failed. Last error: {answers[-1]!r}")
            return
    print()


# --- Export -------------------------------------------------------------------------------------


def simplify(text: str) -> str:
    return " ".join(text.lower().split())


def quote_found(quote: str, text: str) -> bool | None:
    """True if the quote occurs in the post (ignoring case and spacing); None if there is no quote."""
    quote = simplify(quote).strip(" .\"'")
    return quote in simplify(text) if quote else None


def make_table(posts: pl.DataFrame, folder: Path) -> pl.DataFrame:
    """One wide row per coded post: <item>, <item>_evidence and <item>_quote_ok for each item."""
    answers = {record["row"]: record for record in read_jsonl(folder / "results.jsonl")}
    rows = []
    for post in posts.iter_rows(named=True):
        if post["row"] not in answers:
            continue
        answer = answers[post["row"]]
        output = answer["output"]
        row = {"row": post["row"], "ID": post["ID"], "Title": post["Title"]}
        for group, items in output.items():
            if group == "valid":
                continue
            for name, item in items.items():
                row[name] = item["present"]
                row[f"{name}_evidence"] = item["evidence"]
                row[f"{name}_quote_ok"] = quote_found(item["evidence"], post_text(post))
        row["valid"] = output["valid"]["is_job_post"]
        row["valid_reason"] = output["valid"]["reason"]
        row["input_tokens"] = answer["input_tokens"]
        row["output_tokens"] = answer["output_tokens"]
        rows.append(row)
    if not rows:
        return pl.DataFrame(schema={"row": pl.Int64})
    return pl.DataFrame(rows, infer_schema_length=None)


def item_names(table: pl.DataFrame) -> list[str]:
    return [col.removesuffix("_quote_ok") for col in table.columns if col.endswith("_quote_ok")]


def print_summary(table: pl.DataFrame) -> None:
    items = item_names(table)
    summary = pl.DataFrame({
        "item": items,
        "share_coded_1": [table[item].mean() for item in items],
        "quotes_found_in_post": [table[f"{item}_quote_ok"].cast(pl.Boolean).mean() for item in items],
    })
    with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, float_precision=2):
        print(summary)
    print(f"Not a job post: {(table['valid'] == 0).sum()} of {table.height}")
    print(f"Mean tokens per post: {table['input_tokens'].mean():,.0f} in, {table['output_tokens'].mean():,.0f} out")


def average(column: pl.Series, digits: int = 3) -> float:
    """Mean of a column, skipping empty values; 0 when there are none."""
    value = column.cast(pl.Float64).mean()
    return round(value, digits) if isinstance(value, float) else 0.0


def write_metrics(table: pl.DataFrame, path: Path) -> None:
    """Summary numbers for DVC: `dvc exp show` puts them side by side for each experiment."""
    items = item_names(table)
    quote_checks = pl.concat([table[f"{item}_quote_ok"].cast(pl.Boolean) for item in items])
    metrics = {
        "posts_coded": table.height,
        "not_job_post": int((table["valid"] == 0).sum()),
        "quotes_found_in_post": average(quote_checks),
        "mean_input_tokens": average(table["input_tokens"], digits=0),
        "mean_output_tokens": average(table["output_tokens"], digits=0),
        "share_coded_1": {item: average(table[item]) for item in items},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics for {table.height} posts to {path}")


def main() -> None:
    # Windows writes piped output (e.g. under DVC) as cp1252, which can't print Polars' table borders.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, help="code a random sample of N posts (a pilot) instead of all posts")
    parser.add_argument("--api", action="store_true", help="use ANTHROPIC_API_KEY instead of your subscription")
    parser.add_argument("--model", default=MODEL, help="Claude model (default: %(default)s)")
    parser.add_argument("--effort", default=EFFORT, choices=Effort.__args__, help="effort level (default: %(default)s)")
    parser.add_argument("--metrics", type=Path, help="also write summary numbers for these posts to this JSON file")
    args = parser.parse_args()

    all_posts = load_posts()
    posts = all_posts.sample(min(args.n, all_posts.height), seed=42) if args.n else all_posts
    folder = output_folder(args.model, args.effort)
    folder.mkdir(parents=True, exist_ok=True)
    done = {record["row"] for record in read_jsonl(folder / "results.jsonl")}
    todo = posts.filter(~pl.col("row").is_in(list(done))).to_dicts()
    print(f"{args.model}, effort {args.effort}: {posts.height - len(todo):,} posts already coded, {len(todo):,} to go")

    if todo:
        if args.api:
            asyncio.run(code_posts(todo, code_with_api, api_agent(args.model, args.effort), folder))
        else:
            asyncio.run(code_posts(todo, code_with_subscription, subscription_options(args.model, args.effort), folder))

    table = make_table(all_posts, folder)  # every post coded with this setup, also by earlier runs
    if table.height:
        table.write_parquet(folder / "results.parquet")
        print_summary(table)
        print(f"Wrote {table.height:,} coded posts to {folder / 'results.parquet'}")
    asked_for = table.filter(pl.col("row").is_in(posts["row"].to_list()))  # coded posts from this sample
    if args.metrics and asked_for.height:
        write_metrics(asked_for, args.metrics)

    left = posts.height - asked_for.height
    if left:
        sys.exit(f"{left:,} posts are not coded yet. Run the same command again (after a usage limit resets).")


if __name__ == "__main__":
    main()
