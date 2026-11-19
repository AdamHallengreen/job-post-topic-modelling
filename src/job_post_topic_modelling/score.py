import json
import re
import time
from pathlib import Path

import polars as pl
from bertopic import BERTopic
from omegaconf import OmegaConf

from job_post_topic_modelling.utils.find_project_root import find_project_root
from job_post_topic_modelling.utils.miscellaneous import print_params

OUTLIER_TOPIC = -1


def load_claude_codes(path: Path) -> pl.DataFrame:
    """Read Claude's codes, keep the posts it judged to be job posts (valid == 1), and rename ID to post_id."""
    codes = pl.read_parquet(path)
    return codes.filter(pl.col("valid") == 1).with_columns(pl.col("ID").cast(pl.String).alias("post_id")).drop("ID")


def item_names(codes: pl.DataFrame) -> list[str]:
    """The coded items: columns ending in _quote_ok, without that suffix (same rule as annotate_jobnet.py)."""
    return [col.removesuffix("_quote_ok") for col in codes.columns if col.endswith("_quote_ok")]


def post_topics(occurrences: pl.DataFrame) -> pl.DataFrame:
    """The distinct (post_id, predicted_topic) pairs, without the outlier topic -1."""
    return occurrences.filter(pl.col("predicted_topic") != OUTLIER_TOPIC).select("post_id", "predicted_topic").unique()


def best_topic_f1(post_topics: pl.DataFrame, codes: pl.DataFrame, item: str) -> tuple[int | None, float | None]:
    """Find the topic whose posts match Claude's code for `item` best, and return (topic, F1).

    A post is "in" a topic if at least one of its sentences is. Only coded posts count (post_topics is joined to
    codes). For each topic: TP = coded posts in the topic with item == 1, FP = those with item == 0,
    FN = all positives - TP, F1 = 2TP / (2TP + FP + FN). On equal F1 the lowest topic id wins.
    Returns (None, None) when no post has the item or there are no topics.
    """
    positives = codes.filter(pl.col(item) == 1).height
    joined = post_topics.join(codes.select("post_id", item), on="post_id")
    if positives == 0 or joined.is_empty():
        return None, None
    per_topic = joined.group_by("predicted_topic").agg(
        tp=(pl.col(item) == 1).sum(),
        fp=(pl.col(item) == 0).sum(),
    )
    scored = per_topic.with_columns(
        f1=2 * pl.col("tp") / (2 * pl.col("tp") + pl.col("fp") + (positives - pl.col("tp")))
    )
    best = scored.sort(["f1", "predicted_topic"], descending=[True, False]).row(0, named=True)
    return best["predicted_topic"], best["f1"]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, so quotes and sentences compare fairly."""
    return re.sub(r"\s+", " ", text).strip().lower()


def match_quote(quote: str, sentences: list[str]) -> list[int]:
    """Positions in `sentences` that contain the quote, or are contained in it (lowercased, whitespace collapsed).

    The second case covers a quote that spans several sentences. Empty sentences are ignored, because an empty
    string is "contained" in every quote.
    """
    quote = normalize(quote)
    matches = []
    for position, sentence in enumerate(sentences):
        sentence = normalize(sentence)
        if sentence and (quote in sentence or sentence in quote):
            matches.append(position)
    return matches


def evidence_shares(occurrences: pl.DataFrame, codes: pl.DataFrame, items: list[str]) -> dict[str, float]:
    """How well Claude's evidence quotes can be found among the post's sentences, and where they land.

    Every item with value 1 and a non-empty evidence quote counts as one quote. A quote is matched if
    match_quote finds at least one sentence of its post. evidence_unmatched_share is unmatched / all quotes.
    evidence_in_outliers_share is matched quotes whose matched sentences are all in topic -1, divided by matched
    quotes. Posts missing from occurrences are skipped. A share with no quotes to divide is 0.0.
    """
    sentences_by_post = {}
    topics_by_post = {}
    for (post_id,), group in occurrences.group_by("post_id", maintain_order=True):
        sentences_by_post[post_id] = group.get_column("text").to_list()
        topics_by_post[post_id] = group.get_column("predicted_topic").to_list()

    n_quotes = n_unmatched = n_in_outliers = 0
    for post in codes.iter_rows(named=True):
        post_id = post["post_id"]
        if post_id not in sentences_by_post:
            continue
        for item in items:
            quote = post[f"{item}_evidence"]
            if post[item] != 1 or not quote or not quote.strip():
                continue
            n_quotes += 1
            matched = match_quote(quote, sentences_by_post[post_id])
            if not matched:
                n_unmatched += 1
            elif all(topics_by_post[post_id][position] == OUTLIER_TOPIC for position in matched):
                n_in_outliers += 1
    n_matched = n_quotes - n_unmatched
    return {
        "evidence_in_outliers_share": n_in_outliers / n_matched if n_matched else 0.0,
        "evidence_unmatched_share": n_unmatched / n_quotes if n_quotes else 0.0,
    }


def topic_diversity(top_words: list[list[str]], n: int = 10) -> float:
    """Share of distinct words among the top-n words of all topics (1.0 = no word is used by two topics)."""
    words = [word for topic_words in top_words for word in topic_words[:n]]
    return len(set(words)) / len(words) if words else 0.0


def representative_sentences(topic_model: BERTopic, occurrences: pl.DataFrame, topic: int, n: int = 3) -> list[str]:
    """BERTopic's representative docs for the topic; if it has none, n random sentences from the topic (seed=1)."""
    try:
        docs = topic_model.get_representative_docs(topic) or []
    except (AttributeError, TypeError):
        docs = []
    if docs:
        return docs[:n]
    in_topic = occurrences.filter(pl.col("predicted_topic") == topic)
    return in_topic.sample(min(n, in_topic.height), seed=1).get_column("text").to_list()


def topics_overview(
    topic_model: BERTopic, occurrences: pl.DataFrame, best_topics: dict[str, int | None], n_top_words: int
) -> pl.DataFrame:
    """One row per topic (not -1): size, top words, 3 representative sentences and the best-matching item."""
    sizes = occurrences.filter(pl.col("predicted_topic") != OUTLIER_TOPIC).get_column("predicted_topic").value_counts()
    rows = []
    for topic, size in sizes.sort("predicted_topic").iter_rows():
        words = [word for word, _ in topic_model.get_topic(topic)[:n_top_words]]
        matching_items = [item for item, best in best_topics.items() if best == topic]
        rows.append({
            "topic": topic,
            "size": size,
            "top_words": ", ".join(words),
            "representative_sentences": " | ".join(representative_sentences(topic_model, occurrences, topic)),
            "best_matching_item": ", ".join(matching_items),
        })
    return pl.DataFrame(rows, schema=["topic", "size", "top_words", "representative_sentences", "best_matching_item"])


if __name__ == "__main__":
    project_root = Path(find_project_root(__file__))
    output_dir = project_root / "output"
    full_par = OmegaConf.load(project_root / "params.yaml")
    par = full_par.score

    print(f"Starting {Path(__file__).name}")
    start = time.time()
    print_params(full_par)

    # 1. Load the topics per sentence occurrence, Claude's codes and the topic words
    occurrences = pl.read_parquet(output_dir / "predicted_topics_train.parquet")
    codes = load_claude_codes(project_root / par.labels_path)
    codes = codes.filter(pl.col("post_id").is_in(occurrences["post_id"]))  # only posts that were in training
    items = item_names(codes)
    topic_model = BERTopic.load(project_root / "models" / "bertopic_model")

    # 2. Per item: the topic that matches Claude's code best
    metrics = {}
    best_topics = {}
    topics_of_posts = post_topics(occurrences)
    for item in items:
        best_topics[item], metrics[f"f1_{item}"] = best_topic_f1(topics_of_posts, codes, item)
        metrics[f"best_topic_{item}"] = best_topics[item]
    f1_scores = [f1 for f1 in (metrics[f"f1_{item}"] for item in items) if f1 is not None]
    metrics["mean_best_topic_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else None

    # 3. Evidence quotes, outliers and topic quality
    metrics.update(evidence_shares(occurrences, codes, items))
    is_outlier = pl.col("predicted_topic") == OUTLIER_TOPIC
    metrics["outlier_share_unique_sentences"] = occurrences.unique("sentence_id").select(is_outlier.mean()).item()
    metrics["outlier_share_occurrences"] = occurrences.select(is_outlier.mean()).item()
    topic_ids = sorted(set(occurrences["predicted_topic"].to_list()) - {OUTLIER_TOPIC})
    top_words = [[word for word, _ in topic_model.get_topic(topic)] for topic in topic_ids]
    metrics["n_topics"] = len(topic_ids)
    metrics["topic_diversity"] = topic_diversity(top_words, n=par.n_top_words)
    metrics["n_coded_posts"] = codes.height

    # 4. Write the metrics and the topic overview
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics" / "score.json").write_text(json.dumps(metrics, indent=2))
    topics_overview(topic_model, occurrences, best_topics, par.n_top_words).write_csv(
        output_dir / "topics_overview.csv"
    )
    print(f"Finished {Path(__file__).name} in {time.time() - start:.1f} seconds")
