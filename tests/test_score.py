import math

import polars as pl

from job_post_topic_modelling.score import (
    best_topic_f1,
    evidence_shares,
    item_names,
    load_claude_codes,
    match_quote,
    post_topics,
    topic_diversity,
)

CODES = pl.DataFrame({
    "post_id": ["A", "B", "C", "D"],
    "flex": [1, 1, 0, 0],
    "flex_evidence": ["flekstid", "fleksibel arbejdstid", "", ""],
    "AI": [0, 0, 0, 0],
    "AI_evidence": ["", "", "", ""],
})
OCC = pl.DataFrame({
    "post_id": ["A", "A", "B", "B", "C", "D"],
    "text": ["Pædagog", "Vi har flekstid.", "Du får fleksibel", "arbejdstid og frihed.", "God løn.", "Kantine."],
    "label": ["A_title", "A_00", "B_00", "B_01", "C_00", "D_00"],
    "predicted_topic": [2, 1, -1, 1, 1, 3],
})


def test_best_topic_f1():
    topic, f1 = best_topic_f1(post_topics(OCC), CODES, "flex")
    assert topic == 1 and math.isclose(f1, 0.8)  # topic 1 in A, B, C: TP=2, FP=1, FN=0


def test_best_topic_f1_no_positives():
    assert best_topic_f1(post_topics(OCC), CODES, "AI") == (None, None)


def test_best_topic_f1_tie_picks_lowest_topic():
    topics = pl.DataFrame({"post_id": ["A", "A"], "predicted_topic": [5, 3]})
    codes = pl.DataFrame({"post_id": ["A"], "x": [1]})
    assert best_topic_f1(topics, codes, "x") == (3, 1.0)


def test_post_topics_drops_outliers_and_duplicates():
    result = post_topics(OCC.vstack(OCC.head(2))).sort("post_id", "predicted_topic")
    assert result.rows() == [("A", 1), ("A", 2), ("B", 1), ("C", 1), ("D", 3)]


def test_match_quote_inside_and_spanning_sentences():
    assert match_quote("FLEKSTID", ["Pædagog", "Vi har flekstid."]) == [1]
    assert match_quote("fleksibel arbejdstid", ["Du får fleksibel", "arbejdstid og frihed."]) == []
    assert match_quote("Du får fleksibel arbejdstid og frihed.", ["Du får fleksibel", "arbejdstid og frihed."]) == [
        0,
        1,
    ]


def test_match_quote_spanning_sentences():
    assert match_quote("Vi har flekstid. God løn.", ["Vi har flekstid.", "God løn.", "Kantine."]) == [0, 1]


def test_match_quote_collapses_whitespace_and_ignores_empty_sentences():
    assert match_quote("vi  har\nflekstid", ["", "  ", "Vi har flekstid."]) == [2]


def test_evidence_shares():
    shares = evidence_shares(OCC, CODES, ["flex", "AI"])
    assert shares == {"evidence_in_outliers_share": 0.0, "evidence_unmatched_share": 0.5}


def test_evidence_shares_counts_outlier_quotes():
    occ = pl.DataFrame({"post_id": ["A", "A"], "text": ["Vi har flekstid.", "God løn."], "predicted_topic": [-1, 1]})
    codes = pl.DataFrame({"post_id": ["A"], "flex": [1], "flex_evidence": ["flekstid"]})
    shares = evidence_shares(occ, codes, ["flex"])
    assert shares == {"evidence_in_outliers_share": 1.0, "evidence_unmatched_share": 0.0}


def test_topic_diversity():
    assert topic_diversity([["a", "b"], ["b", "c"]], n=2) == 0.75


def test_load_claude_codes_and_item_names(tmp_path):
    table = pl.DataFrame({
        "ID": [1, 2],
        "valid": [1, 0],
        "flex": [1, 1],
        "flex_evidence": ["x", "y"],
        "flex_quote_ok": [1, 1],
    })
    table.write_parquet(tmp_path / "results.parquet")
    codes = load_claude_codes(tmp_path / "results.parquet")
    assert codes["post_id"].to_list() == ["1"]
    assert item_names(codes) == ["flex"]
