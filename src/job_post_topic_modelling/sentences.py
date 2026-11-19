"""Pure text helpers: clean HTML, split job posts into sentences, spot headings and sensitive text."""

import os
import re

import nltk  # type: ignore  # noqa: PGH003
import polars as pl
from nltk.tokenize import sent_tokenize

# Check if running on STAR server, if yes set up path to load nltk data
user = os.popen("whoami").read().strip()  # noqa: S605, S607
nltk.data.path.append(rf"/home/{user}@PROD.SITAD.DK/code/help/installations/nltk_data")

# Tags that end a line: <br>, <li>, </li>, </p>, </div> (any case, with attributes)
LINE_BREAK_TAG = re.compile(r"<\s*(?:br|li|/li|/p|/div)\b[^>]*>", re.IGNORECASE)
# Any other HTML tag
ANY_TAG = re.compile(r"<[^>]+>")
# Runs of spaces, tabs and non-breaking spaces
EXTRA_SPACES = re.compile(r"[ \t\xa0]+")
# Sentence mode: every line break, bullet or asterisk starts a new part
SENTENCE_SEPARATOR = re.compile(r"\r?\n|•|\*")
# Paragraph mode: only blank lines, bullets or asterisks start a new part
PARAGRAPH_SEPARATOR = re.compile(r"(?:\r?\n){2,}|•|\*")
# Any whitespace run, collapsed to one space
WHITESPACE = re.compile(r"\s+")

# Sentences that are only a section title, compared in lower case without trailing :?.!
KNOWN_HEADINGS = {
    "om opgaven",
    "om jobbet",
    "om stillingen",
    "stillingen",
    "om dig",
    "om os",
    "om virksomheden",
    "din profil",
    "dine opgaver",
    "arbejdsopgaver",
    "kvalifikationer",
    "faglige kvalifikationer",
    "personlige kvalifikationer",
    "vi tilbyder",
    "vi tilbyder dig",
    "vi forventer",
    "vi søger",
    "hvem er du",
    "løn og ansættelsesvilkår",
    "ansættelsesvilkår",
    "ansøgning",
    "yderligere oplysninger",
    "kontakt",
    "praktiske oplysninger",
}
MAX_WORDS_IN_QUESTION_HEADING = 4

# Sensitive text: dates, phone numbers, e-mail addresses and homepages
DATE_PATTERN = re.compile(
    r"\b(\d{1,2}([./-])\d{1,2}\2\d{2,4}"
    r"|\d{1,2}\.\s*(jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec|januar|februar|marts|april|maj|juni|juli|august|september|oktober|november|december)[a-zæøå]*\s*\d{4})\b",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(r"\b(\d{8}|(\d{2}\s){3}\d{2})\b")
# Danish phone number written as two groups of four digits
PHONE_FOUR_FOUR_PATTERN = re.compile(r"\b\d{4}\s\d{4}\b")
# Phone number with the +45 country code
PHONE_COUNTRY_CODE_PATTERN = re.compile(r"\+45[\s\d]{8,}")
EMAIL_PATTERN = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
HOMEPAGE_PATTERN = re.compile(
    r"http[s]?://\S+|www\.\S+|\b[\w\.-]+\.(dk|com|net|org|info|eu|io|co|biz|gov|edu)\b", re.IGNORECASE
)
SENSITIVE_PATTERNS = [
    DATE_PATTERN,
    PHONE_PATTERN,
    PHONE_FOUR_FOUR_PATTERN,
    PHONE_COUNTRY_CODE_PATTERN,
    EMAIL_PATTERN,
    HOMEPAGE_PATTERN,
]


def post_id_from_section_id(section_ids: pl.Expr) -> pl.Expr:
    """Post id of a STAR section id: "12345_s2" -> "12345"."""
    return section_ids.str.extract(r"^(\d+)_s", 1)


def html_to_text(text: str) -> str:
    """Turn HTML into plain text where list items and paragraphs are separate lines."""
    text = LINE_BREAK_TAG.sub("\n", text)
    text = ANY_TAG.sub(" ", text)
    text = text.replace("_x0009_", " ")
    text = EXTRA_SPACES.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def split_into_sentences(text: str, unit: str = "sentence") -> list[str]:
    """Split text into sentences (unit="sentence") or blank-line blocks (unit="paragraph")."""
    if unit == "paragraph":
        return _split_into_paragraphs(text)
    return _split_into_single_sentences(text)


def _split_into_single_sentences(text: str) -> list[str]:
    """Split on line breaks and bullets, then split each part into sentences."""
    sentences = []
    for part in SENTENCE_SEPARATOR.split(text):
        sentences.extend(sent_tokenize(part, language="danish"))
    return _clean_and_drop_empty(sentences)


def _split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines and bullets, and keep the sentences of one block together."""
    paragraphs = []
    for block in PARAGRAPH_SEPARATOR.split(text):
        sentences = _clean_and_drop_empty(sent_tokenize(block, language="danish"))
        paragraphs.append(" ".join(sentences))
    return _clean_and_drop_empty(paragraphs)


def _clean_and_drop_empty(parts: list[str]) -> list[str]:
    """Collapse inner whitespace and remove empty strings."""
    cleaned = [WHITESPACE.sub(" ", part).strip() for part in parts]
    return [part for part in cleaned if part]


def is_heading(sentence: str) -> bool:
    """True if the sentence is a section title and not content."""
    sentence = sentence.strip()
    if sentence.endswith(":"):
        return True
    if sentence.endswith("?") and len(sentence.split()) <= MAX_WORDS_IN_QUESTION_HEADING:
        return True
    return sentence.lower().rstrip(":?.!") in KNOWN_HEADINGS


def is_sensitive(sentence: str) -> bool:
    """True if the sentence contains a date, phone number, e-mail address or homepage."""
    return any(pattern.search(sentence) for pattern in SENSITIVE_PATTERNS)


def split_posts(posts: pl.DataFrame, unit: str = "sentence") -> pl.DataFrame:
    """Split each post into parts, one row per part.

    Input columns: post_id, text, and optionally label_base. Output columns: post_id, label, text, drop_reason.
    label is "<label_base>_<i>" with i counted from 00 over the row's parts, in order. label_base defaults to
    post_id; use it when one post is spread over several rows (STAR sections), so labels stay unique.
    The output always keeps the real post_id.
    drop_reason is null for parts to keep, "heading" for section titles and "sensitive" for dates,
    phone numbers, e-mail addresses and homepages.
    """
    if "label_base" not in posts.columns:
        posts = posts.with_columns(pl.col("post_id").alias("label_base"))
    rows = []
    for post_id, label_base, text in posts.select("post_id", "label_base", "text").iter_rows():
        clean_text = html_to_text(text or "")
        for i, part in enumerate(split_into_sentences(clean_text, unit)):
            rows.append({
                "post_id": post_id,
                "label": f"{label_base}_{i:02d}",
                "text": part,
                "drop_reason": _drop_reason(part),
            })
    schema = {"post_id": posts.schema["post_id"], "label": pl.String, "text": pl.String, "drop_reason": pl.String}
    return pl.DataFrame(rows, schema=schema)


def _drop_reason(sentence: str) -> str | None:
    """Return "heading", "sensitive" or None (keep). Headings are checked first."""
    if is_heading(sentence):
        return "heading"
    if is_sensitive(sentence):
        return "sensitive"
    return None


def title_rows(posts: pl.DataFrame) -> pl.DataFrame:
    """One row per post with a non-empty title. Input columns: post_id, title.

    Output has the same columns as split_posts, with label "<post_id>_title".
    Titles are never checked with is_heading, a short title is still a title.
    """
    titles = posts.select("post_id", pl.col("title").str.strip_chars().alias("text"))
    titles = titles.filter(pl.col("text").is_not_null() & (pl.col("text") != ""))
    return titles.select(
        "post_id",
        pl.format("{}_title", pl.col("post_id")).alias("label"),
        "text",
        pl.lit(None, dtype=pl.String).alias("drop_reason"),
    )


def deduplicate(texts: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Give every distinct text one sentence_id, so each distinct text is embedded only once.

    Returns (texts_with_sentence_id, sentences):
    - sentences has one row per distinct text: sentence_id (0, 1, 2, ... in order of first appearance),
      text, and n_occurrences (how many rows of `texts` have this text).
    - texts_with_sentence_id is `texts` unchanged, same row order, plus a sentence_id column.
      sentence_id links each row to its row in `sentences`.
    """
    # Step 1: the distinct texts, in order of first appearance
    sentences = texts.select("text").unique(maintain_order=True)

    # Step 2: number them 0, 1, 2, ...
    sentences = sentences.with_row_index("sentence_id")

    # Step 3: count how often each text occurs in the input
    counts = texts.group_by("text").agg(pl.len().alias("n_occurrences"))
    sentences = sentences.join(counts, on="text", how="left", maintain_order="left")

    # Step 4: look up each input row's sentence_id by its text, keeping the input row order
    texts_with_sentence_id = texts.join(
        sentences.select("sentence_id", "text"), on="text", how="left", maintain_order="left"
    )
    return texts_with_sentence_id, sentences.select("sentence_id", "text", "n_occurrences")
