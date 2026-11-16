# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.32.2"]
# ///
"""Compare how different Claude models coded the same posts.

Run it after the pilot experiments (see README.md in this folder):

    uv run scripts/llm_topics/compare_pilots.py

It reads results.parquet from every folder in data/llm_topics made with the current prompt.txt, keeps
the posts that all of them coded, and prints for each item how often the models agree. The posts where
they disagree go to data/llm_topics/pilot_disagreements.csv, to read by hand (opens in Excel).
"""

import hashlib
import sys
from pathlib import Path

import polars as pl

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parents[1] / "data" / "llm_topics"
# Same prompt id as in annotate_jobnet.py: results made with an older prompt.txt are left out.
PROMPT = (SCRIPT_DIR / "prompt.txt").read_text(encoding="utf-8").strip()
PROMPT_ID = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:8]


def main() -> None:
    # One table per model, e.g. {"claude-haiku-5-5_low": <results>, ...}
    tables = {
        folder.name.removesuffix(f"_prompt-{PROMPT_ID}"): pl.read_parquet(folder / "results.parquet")
        for folder in sorted(RESULTS_DIR.glob(f"*_prompt-{PROMPT_ID}"))
        if (folder / "results.parquet").exists()
    }
    if len(tables) < 2:
        sys.exit(f"Found results for {len(tables)} model(s) with the current prompt.txt; need at least 2.")
    models = list(tables)

    # Keep the posts every model coded, and put the models side by side: one column per model and item.
    common_rows = set.intersection(*(set(table["row"]) for table in tables.values()))
    first = tables[models[0]]
    items = [col.removesuffix("_quote_ok") for col in first.columns if col.endswith("_quote_ok")]
    wide = first.filter(pl.col("row").is_in(common_rows)).select("row", "Title")
    for model, table in tables.items():
        columns = [pl.col(item).alias(f"{item} {model}") for item in [*items, "valid"]]
        columns += [pl.col(f"{item}_evidence").alias(f"{item}_evidence {model}") for item in items]
        wide = wide.join(table.select("row", *columns), on="row")
    print(f"Comparing {', '.join(models)} on the {wide.height} posts they all coded\n")

    # Per item: share coded 1 by each model, and how often all models give the same code.
    summary = []
    for item in [*items, "valid"]:
        codes = [pl.col(f"{item} {model}") for model in models]
        same = wide.select(pl.all_horizontal(code == codes[0] for code in codes[1:])).to_series()
        any_1 = wide.select(pl.any_horizontal(code == 1 for code in codes)).to_series()
        row: dict[str, object] = {"item": item}
        for model in models:
            row[f"share_1 {model}"] = wide[f"{item} {model}"].mean()
        row["all_agree"] = same.mean()
        # Agreement is high for rare items just because most codes are 0, so also look at the posts
        # where at least one model said 1: how often did all of them say 1?
        row["all_agree_when_any_1"] = same.filter(any_1).mean()
        summary.append(row)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(summary))

    # The disagreements, one row per post and item, with each model's code and quote.
    disagreements = []
    for item in items:
        codes = [pl.col(f"{item} {model}") for model in models]
        disagree = wide.filter(~pl.all_horizontal(code == codes[0] for code in codes[1:]))
        disagreements.append(
            disagree.select(
                "row",
                "Title",
                pl.lit(item).alias("item"),
                *[pl.col(f"{item} {model}").alias(f"code {model}") for model in models],
                *[pl.col(f"{item}_evidence {model}").alias(f"quote {model}") for model in models],
            )
        )
    out = RESULTS_DIR / "pilot_disagreements.csv"
    table = pl.concat(disagreements).sort("row", "item")
    table.write_csv(out, include_bom=True)
    print(f"\nWrote {table.height} disagreements to {out}")


if __name__ == "__main__":
    main()
