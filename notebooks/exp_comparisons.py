import ast
import subprocess

import polars as pl
from dvc.api import open as dvc_open  # works across DVC 2/3

# Path to your YAML artifact relative to repo root
ARTIFACT = "output/topic_info.csv"


def _list_experiment_refs() -> list[dict[str, str]]:
    """
    Return a list of dicts with keys: ref, label, sha.
    Uses pure Git to enumerate DVC experiment refs.
    """
    out = subprocess.check_output(
        ["git", "for-each-ref", "refs/exps", "--format=%(refname:short) %(objectname)"],  # noqa: S607
        text=True,
    )
    exps = []
    for line in out.strip().splitlines():
        ref, sha = line.split()
        label = ref.split("/")[-1]  # short tail of the ref; not always pretty, but stable
        exps.append({"ref": ref, "label": label, "sha": sha})
    return exps


def _match_experiment(exps: list[dict[str, str]], query: str) -> dict[str, str] | None:
    """
    Match by:
      - exact SHA
      - SHA prefix
      - exact label
      - case-insensitive label contains (if unique)
    Returns the single matched experiment dict or None.
    If multiple fuzzy matches, prints choices and returns None.
    """
    query_l = query.lower()

    # exact sha or exact label
    exact = [e for e in exps if e["sha"] == query or e["label"] == query]
    if exact:
        return exact[0]

    # sha prefix
    pref = [e for e in exps if e["sha"].startswith(query)]
    if len(pref) == 1:
        return pref[0]
    if len(pref) > 1:
        print(f"Multiple experiments match SHA prefix '{query}':")
        for e in pref:
            print(f"  {e['label']}  {e['sha'][:7]}")
        return None

    # label contains (case-insensitive)
    fuzzy = [e for e in exps if query_l in e["label"].lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        print(f"Multiple experiments match label '{query}':")
        for e in fuzzy:
            print(f"  {e['label']}  {e['sha'][:7]}")
        return None

    print(f"No experiment matched '{query}'.")
    return None


def _load_csv_at_ref(path: str, rev: str):
    with dvc_open(path, rev=rev, mode="r") as f:
        return pl.read_csv(f).with_columns(
            pl.col("Representation").map_elements(ast.literal_eval, return_dtype=pl.List(pl.Utf8))
        )


def print_all_experiments_text(artifact: str = ARTIFACT):
    """
    Print the csv artifact for all experiments as plain text.
    """
    exps = _list_experiment_refs()
    if not exps:
        print("No DVC experiments found (refs/exps/*).")
        return

    for i, e in enumerate(exps, 1):
        print("=" * 80)
        print(f"[{i}/{len(exps)}] EXPERIMENT: {e['label']}  ({e['sha'][:12]})")
        print("-" * 80)
        try:
            data = _load_csv_at_ref(artifact, rev=e["sha"])
            print(data.head(10))

        except Exception as err:
            print(f"[ERROR] Could not load {artifact} for {e['label']} ({e['sha'][:7]}): {err}")
        print()  # blank line after each


def load_experiment_csv(exp_query: str, artifact: str = ARTIFACT, return_data=True, n_words=10, n_topics=10):
    """
    Print the csv artifact for a specific experiment identified by:
      - experiment name/label (from the ref tail),
      - SHA,
      - or SHA prefix.
    """
    exps = _list_experiment_refs()
    if not exps:
        print("No DVC experiments found (refs/exps/*).")
        return

    match = _match_experiment(exps, exp_query)
    if not match:
        # already printed a helpful message
        return

    print("=" * 80)
    print(f"EXPERIMENT: {match['label']}  ({match['sha'][:12]})")
    print("-" * 80)
    try:
        data = _load_csv_at_ref(artifact, rev=match["sha"])
        print_top_topics(data, n_topics=5)

    except Exception as err:
        print(f"[ERROR] Could not load {artifact}: {err}")
        return_data = False

    if return_data:
        return data


def print_top_topics(data, n_words=10, n_topics=10):
    # Filter out outliers (-1), sort by count, take top 10
    topn = data.filter(pl.col("Topic") != -1).sort("Count", descending=True)
    if n_topics is not None:
        topn = topn.head(n_topics)

    # Iterate and print
    for row in topn.iter_rows(named=True):
        topic_id = row["Topic"]
        count = row["Count"]
        name = row["Name"]
        rep = row["Representation"]  # now a list of (word, score) pairs
        words = rep[:n_words]
        print(f"Topic {topic_id + 1} ({count} docs): {name}")
        print("  Top words:", ", ".join(words))
        print()


def print_top_topics_from_query(query, artifact: str = ARTIFACT, n_words=10, n_topics=10):
    print(f"Query: {query}")
    data = load_experiment_csv(query, artifact=artifact)
    print_top_topics(data)


def print_experiments(exp_queries: list[str], artifact: str = ARTIFACT):
    """
    Print the YAML artifacts for multiple experiments.
    """
    for query in exp_queries:
        print_top_topics_from_query(query, artifact=artifact)
        print()  # blank line after each
