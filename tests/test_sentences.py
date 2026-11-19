import polars as pl

from job_post_topic_modelling.sentences import (
    deduplicate,
    html_to_text,
    is_heading,
    is_sensitive,
    post_id_from_section_id,
    split_into_sentences,
    split_posts,
    title_rows,
)


def test_html_list_tags_become_line_breaks():
    assert html_to_text("<ul><li>Løn</li><li>Pension</li></ul>").split("\n") == ["Løn", "Pension"]


def test_html_removes_x0009_and_extra_spaces_but_keeps_line_breaks():
    assert html_to_text("Vi  har_x0009_flekstid.\nGod løn.") == "Vi har flekstid.\nGod løn."


def test_single_line_break_list_is_split():
    text = "Du er struktureret\nDu taler engelsk\nDu har kørekort"
    assert split_into_sentences(text) == ["Du er struktureret", "Du taler engelsk", "Du har kørekort"]


def test_bullets_and_sentences_are_split():
    assert split_into_sentences("Vi tilbyder flekstid. God pension.•Frokostordning") == [
        "Vi tilbyder flekstid.",
        "God pension.",
        "Frokostordning",
    ]


def test_paragraph_unit_keeps_lines_within_a_blank_line_block_together():
    assert split_into_sentences("Linje et.\nLinje to.\n\nNy blok.", unit="paragraph") == [
        "Linje et. Linje to.",
        "Ny blok.",
    ]


def test_headings_are_detected():
    for heading in ["Vi tilbyder:", "Hvem er du?", "Om opgaven", "Din profil", "Dine arbejdsopgaver inkluderer:"]:
        assert is_heading(heading)


def test_short_sentences_with_content_are_kept():
    for sentence in [
        "Vi har flekstid.",
        "fastansættelse.",
        "Grib chancen nu!",
        "Løn efter kvalifikationer.",
        "Frokostordning",
        "Fleksible arbejdstider",
        "Du taler engelsk",
        "Har du lyst til at arbejde med børn og unge?",
    ]:
        assert not is_heading(sentence)


def test_phone_formats_are_sensitive():
    assert is_sensitive("Ring på 3634 4990")
    assert is_sensitive("Ring på +45 36 34 49 90")
    assert is_sensitive("Ring på 36344990")
    assert not is_sensitive("Vi har 2024 ansatte og 1200 kunder")


def test_split_posts_marks_drop_reasons_and_labels():
    posts = pl.DataFrame({"post_id": ["p1"], "text": ["Vi tilbyder:\nFlekstid.\nRing på 3634 4990."]})
    out = split_posts(posts)
    assert out["label"].to_list() == ["p1_00", "p1_01", "p1_02"]
    assert out["drop_reason"].to_list() == ["heading", None, "sensitive"]


def test_title_rows_skip_empty_titles():
    posts = pl.DataFrame({"post_id": ["p1", "p2", "p3"], "title": ["Pædagog", None, " "]})
    out = title_rows(posts)
    assert out.select("post_id", "label", "text").rows() == [("p1", "p1_title", "Pædagog")]


def test_deduplicate_round_trip():
    texts = pl.DataFrame({
        "post_id": ["a", "a", "b"],
        "label": ["a_00", "a_01", "b_00"],
        "text": ["Vi tilbyder flekstid.", "God løn.", "Vi tilbyder flekstid."],
    })
    texts_out, sentences = deduplicate(texts)
    assert sentences.rows() == [(0, "Vi tilbyder flekstid.", 2), (1, "God løn.", 1)]
    assert texts_out["sentence_id"].to_list() == [0, 1, 0]
    rebuilt = texts_out.join(sentences, on="sentence_id", suffix="_unique")
    assert (rebuilt["text"] == rebuilt["text_unique"]).all()


def test_post_id_from_section_id():
    df = pl.DataFrame({"section_id": ["12345_s1", "12345_s2", "987_s10"]})
    out = df.select(post_id_from_section_id(pl.col("section_id")))["section_id"].to_list()
    assert out == ["12345", "12345", "987"]


def test_split_posts_label_base_keeps_real_post_id():
    posts = pl.DataFrame({"post_id": ["1"], "label_base": ["1_s2"], "text": ["Flekstid. God løn."]})
    out = split_posts(posts)
    assert out["post_id"].to_list() == ["1", "1"]
    assert out["label"].to_list() == ["1_s2_00", "1_s2_01"]
