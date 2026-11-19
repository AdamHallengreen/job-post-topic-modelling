"""Pure text helpers: clean HTML, split job posts into sentences, spot headings and sensitive text."""

import os
import re

import nltk  # type: ignore  # noqa: PGH003
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
