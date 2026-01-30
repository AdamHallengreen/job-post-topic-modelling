import json
import os
import re
import time
from pathlib import Path

# For plotting
import polars as pl
from lingua import LanguageDetectorBuilder
from nltk.tokenize import sent_tokenize
from omegaconf import DictConfig, OmegaConf
from polars import col as c

from job_post_topic_modelling.utils.miscellaneous import print_params, try_inter

try_inter()
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402

# Check if running on STAR server, if yes set up path to load nltk data

user = os.popen("whoami").read().strip()  # noqa: S605, S607
import nltk  # type: ignore  # noqa: PGH003 E402

nltk.data.path.append(rf"/home/{user}@PROD.SITAD.DK/code/help/installations/nltk_data")


class FileNotFoundErrorMessage:
    def __init__(self, file_path: Path) -> None:
        self.message = f"File {file_path} does not exist."

    def __str__(self) -> str:
        return self.message


class UnsupportedFileTypeError(Exception):
    def __init__(self, file_suffix: str) -> None:
        self.message = f"Unsupported file type: {file_suffix}"
        super().__init__(self.message)


class UnsupportedLinguaOutput(Exception):
    def __init__(self, output: str) -> None:
        self.message = f"Unsupported Lingua output: {output}"
        super().__init__(self.message)


def load_data(file_path: Path, par: DictConfig) -> pl.DataFrame:
    """
    Load data from an Excel file.

    Args:
        file_path (Path): Path to the Excel file.
        par (DictConfig): Configuration parameters.

    Returns:
        pl.DataFrame: DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnsupportedFileTypeError: If the file is not an Excel file.
    """
    # Check if using STAR data
    if par.star.usestar == 1:
        return load_star_data(par)

    # check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(FileNotFoundErrorMessage(file_path))
    # check if the file is an Excel file
    if file_path.suffix in [".xlsx", ".xls"]:
        df = load_excel(file_path)
    else:
        raise UnsupportedFileTypeError(file_path.suffix)
    return df


def load_star_data(par) -> pl.DataFrame:
    """
    Loads the jobpost data from star data on the server
    """

    folder_path = Path("/data/projects/klikdata/Asker/jobads/dgp/textdata")

    dataname = "jobads_clean.parquet"
    id_var = "ann_id"
    text_var = "annonce_tekst"
    if par.settings.split_sentences or par.settings.split_paragraphs:
        # Load the sectioned data
        dataname = "jobads_sections_clean.parquet"
        id_var = "section_id"
        text_var = "section_text"

    df = (
        pl.scan_parquet(folder_path / dataname)
        .select(c(id_var).alias("id"), c(text_var).alias("text"))
        .filter(
            c("text").is_not_null()  # a few obs have missing text but non-missing heading and rubrik
        )
        .slice(0, par.settings.nobs)
    ).collect()

    return df


def rename_jobcenter_obs(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rename rows with id 'Virksomheden har valgt at rekruttere via jobcentret' to unique IDs.

    Args:
        df (pl.DataFrame): Input DataFrame with an 'id' column.

    Returns:
        pl.DataFrame: DataFrame with unique IDs for jobcenter posts.
    """
    # Add a unique suffix to each duplicate "jobcenter" id using cumcount
    df = df.with_columns(
        pl.when(c("id") == "Virksomheden har valgt at rekruttere via jobcentret")
        .then("jobcenter_" + (pl.cum_count("id").over("id") + 1).cast(pl.String))
        .otherwise(c("id"))
        .alias("id")
    )
    return df


def load_excel(file_path: Path, sheet_name: str = "Sheet1") -> pl.DataFrame:
    """
    Load data from an Excel file and return it as a DataFrame.

    Args:
        file_path (Path): Path to the Excel file.
        sheet_name (str, optional): Name of the sheet to load. Defaults to "Sheet1".

    Returns:
        pl.DataFrame: DataFrame representing the rows in the Excel file.
    """

    df = (
        pl.read_excel(
            file_path,
            sheet_name=sheet_name,
            columns=["ID", "Text"],
            schema_overrides={"ID": pl.String, "Text": pl.String},
        )
        .rename({"ID": "id", "Text": "text"})
        .slice(0, par.settings.nobs)
    )
    return df


def detect_language(df: pl.DataFrame) -> pl.DataFrame:
    """
    Detect the language of the texts using Lingua and add a 'language' column.

    Args:
        df (pl.DataFrame): DataFrame with a 'text' column.

    Returns:
        pl.DataFrame: DataFrame with an added 'language' column.
    """
    detector = LanguageDetectorBuilder.from_all_languages().build()
    languages = []
    texts = df.select(c("text")).to_series().to_list()

    # We could possibly speed this up by setting the languages to detect
    outputs = detector.detect_languages_in_parallel_of(texts)
    for output in outputs:
        if output is not None:
            if hasattr(output, "iso_code_639_3"):
                languages.append(output.iso_code_639_3.name)
            else:
                raise UnsupportedLinguaOutput(str(output))
        else:
            languages.append("unknown")

    df = df.with_columns(pl.Series("language", languages))
    return df


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean the DataFrame by renaming jobcenter posts and filtering for Danish language.

    Args:
        df (pl.DataFrame): Input DataFrame.

    Returns:
        pl.DataFrame: Cleaned DataFrame.
    """
    # rename job center posts
    df = rename_jobcenter_obs(df)

    # remove non-danish posts
    df = detect_language(df)
    n = df.height
    df = df.filter(c("language") == "DAN")
    print(f"    - Removed {n - df.height:,} non-danish texts")

    # clean text column
    df = df.with_columns(
        c("text")
        # Remove HTML tags
        .str.replace_all(r"<[^>]+>", " ")
        # Remove _x0009_
        .str.replace_all(r"_x0009_", " ")
        # Remove extra tabs and spaces except line breaks
        .str.replace_all(r"[ \t]+", " ")
        # Remove leading and trailing whitespace
        .str.strip_chars()
        .alias("text")
    )

    return df


def filter_sentences_remove_sensitive(df: pl.DataFrame, split_paragraphs: bool = False) -> tuple[pl.DataFrame, list]:
    """
    Splits texts into sentences or paragraphs, removes those containing sensitive info,
    and returns a DataFrame with unique labels and filtered text.
    """
    # Compile regex patterns for sensitive info
    date_pattern = re.compile(
        r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
        r"|\d{1,2}\.\s*(jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec|januar|februar|marts|april|maj|juni|juli|august|september|oktober|november|december)[a-zæøå]*\s*\d{4})\b",
        re.IGNORECASE,
    )
    date_pattern = re.compile(
        r"\b(\d{1,2}([./-])\d{1,2}\2\d{2,4}"
        + r"|\d{1,2}\.\s*(jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec|januar|februar|marts|april|maj|juni|juli|august|september|oktober|november|december)[a-zæøå]*\s*\d{4})\b",
        re.IGNORECASE,
    )
    phone_pattern = re.compile(r"\b(\d{8}|(\d{2}\s){3}\d{2})\b")
    email_pattern = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
    homepage_pattern = re.compile(
        r"http[s]?://\S+|www\.\S+|\b[\w\.-]+\.(dk|com|net|org|info|eu|io|co|biz|gov|edu)\b", re.IGNORECASE
    )

    def is_sensitive(text: str) -> bool:
        return (
            date_pattern.search(text)
            or phone_pattern.search(text)
            or email_pattern.search(text)
            or homepage_pattern.search(text)
        )

    labels, filtered_texts = [], []
    sensitive_texts = []
    label_col, text_col = df.columns[0], df.columns[1]
    n_text = df.height
    n_paragraphs = 0
    n_sentences = 0
    n_filtered = 0

    for label, text in zip(df[label_col], df[text_col]):
        idx = 0
        text = str(text)
        # paragraphs = re.split(r"(?:\r?\n){2,}|•", text) if split_paragraphs else [text]
        # always split into paragraphs as it helps with sentences splitting lists. (which explitly (replace html marks with this) use • as separator)
        paragraphs = re.split(r"(?:\r?\n){2,}|•|\*", text)
        n_paragraphs += len(paragraphs)
        for para in paragraphs:
            sentences = sent_tokenize(para, language="danish")
            n_sentences += len(sentences)
            filtered_sents = [re.sub(r"[\r\n]+", " ", sent).strip() for sent in sentences if not is_sensitive(sent)]
            n_filtered += len(sentences) - len(filtered_sents)
            if split_paragraphs:
                if filtered_sents:
                    labels.append(f"{label}_{idx:02d}")
                    filtered_texts.append(" ".join(filtered_sents))
                    idx += 1
            else:
                for sent_clean in filtered_sents:
                    labels.append(f"{label}_{idx:02d}")
                    filtered_texts.append(sent_clean)
                    idx += 1
            sensitive_texts += [sent for sent in sentences if is_sensitive(sent)]

    print(f"    - From {n_text:,} texts, extracted {n_paragraphs:,} paragraphs and {n_sentences:,} sentences")
    print(f"    - Removed {n_filtered:,} sensitive {'paragraphs' if split_paragraphs else 'sentences'}")

    return pl.DataFrame({"label": labels, "text": filtered_texts}), sensitive_texts


def export_texts(texts: pl.DataFrame, output_file: Path) -> None:
    """
    Export the texts to a JSON file.
    Args:
        texts (pl.DataFrame): DataFrame containing the texts.
        output_file (Path): Path to the output file.
    """
    texts.write_parquet(output_file)


def load_texts(file_path: Path) -> pl.DataFrame:
    """
    Load texts from a Parquet file.

    Args:
        file_path (Path): Path to the Parquet file.

    Returns:
        pl.DataFrame: DataFrame containing the texts.
    """
    if not file_path.exists():
        raise FileNotFoundError(FileNotFoundErrorMessage(file_path))
    return pl.read_parquet(file_path)


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    params_path = project_root / "params.yaml"
    data_dir = project_root / "data"
    output_dir = project_root / "output"

    file_path = data_dir / "Jobnet.xlsx"
    texts_file = data_dir / "texts.parquet"

    # Load parameters
    full_par = OmegaConf.load(params_path)
    par = full_par.prepare

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()
    print_params(full_par)

    # Load
    print("Loading data...")
    texts = load_data(file_path, par)
    num_texts_loaded = len(texts)

    # Clean
    print("Cleaning data...")
    texts = clean_data(texts)
    num_texts_used = len(texts)
    from_path = file_path
    if par.star.usestar == 1:
        from_path = Path("STAR data")
    print(f"    - Use {num_texts_used:,}/{num_texts_loaded:,} texts from {from_path}")

    # Split
    if par.settings.split_sentences:
        split_paragraphs = False  # split on sentences
        texts, sensitive_texts = filter_sentences_remove_sensitive(texts, split_paragraphs=split_paragraphs)
        num_splitted_sections = len(texts)
        print(f"    - Split {num_texts_used:,} texts into {num_splitted_sections:,} sentences")
    elif par.settings.split_paragraphs:
        split_paragraphs = True  # split on paragraphs
        texts, sensitive_texts = filter_sentences_remove_sensitive(texts, split_paragraphs=split_paragraphs)
        num_splitted_sections = len(texts)
        print(f"    - Split {num_texts_used:,} texts into {num_splitted_sections:,} paragraphs")
    else:
        pass

    # Save
    export_texts(texts, texts_file)
    print(f"Exported texts to {texts_file}")

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Save metrics
    with ((output_dir / "metrics") / "prepare.json").open("w") as f:
        json.dump({f"{Path(__file__).name}": f"{hours:.2f} hours"}, f, indent=4)

    if False:  # For debugging purposes
        text_org = load_data(file_path, par)
        texts = pl.read_parquet(texts_file)
        print("Sensitive texts:")
        import random

        for st in random.sample(sensitive_texts, 100):
            print(f" - {st}")

        print("Sample original and filtered texts:")
        ids = text_org.select(c.id.str.replace_all(r"_s\d+$", "")).unique().sort("id")
        for ann_id in ids.sample(100)["id"].to_list():
            print(ann_id)
            text_ids = text_org.filter(c("id").str.starts_with(ann_id))
            print(f"Original has {len(text_ids)} parts:")
            print(f"Filtered has {len(texts.filter(c('label').str.starts_with(ann_id)))} parts:")
            for sub_id in text_ids["id"].to_list():
                print(sub_id)
                text_org_obs = text_org.filter(c("id") == sub_id)["text"][0]
                print(text_org_obs)
                filtered_obs = texts.filter(c("label").str.starts_with(sub_id + "_"))

                if len(filtered_obs) == 0:
                    print("   - NO FILTERED TEXTS")
                else:
                    print(f"Has {len(text_org_obs)} characters and {len(filtered_obs)} filtered parts")
                    print("Filtered texts:")
                    for t in filtered_obs["text"].to_list():
                        print(f"   - {t}")
            print("-----\n")
