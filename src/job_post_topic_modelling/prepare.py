import json
import time
from pathlib import Path

import polars as pl
from lingua import LanguageDetectorBuilder
from omegaconf import DictConfig, OmegaConf
from polars import col as c

from job_post_topic_modelling.sentences import deduplicate, post_id_from_section_id, split_posts, title_rows
from job_post_topic_modelling.utils.miscellaneous import print_params, try_inter

try_inter()
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402


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
    Load data from an Excel file, or from STAR when par.star.usestar is 1.

    Args:
        file_path (Path): Path to the Excel file.
        par (DictConfig): Configuration parameters.

    Returns:
        pl.DataFrame: DataFrame with post_id, title and text (and section_id on STAR sections).

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
        df = load_excel(file_path, par)
    else:
        raise UnsupportedFileTypeError(file_path.suffix)
    return df


def load_star_data(par: DictConfig) -> pl.DataFrame:
    """
    Load the job post data from STAR.

    Returns post_id, title, text, plus section_id when the sectioned data is used.
    post_id is ann_id if the file has it, otherwise it is taken from section_id.
    title comes from par.star.title_col and is null when that is not set.
    """
    folder_path = Path("/data/projects/klikdata/Asker/jobads/dgp/textdata")

    sectioned = par.settings.split_sentences or par.settings.split_paragraphs
    if sectioned:
        dataname, text_var = "jobads_sections_clean.parquet", "section_text"
    else:
        dataname, text_var = "jobads_clean.parquet", "annonce_tekst"

    lazy = pl.scan_parquet(folder_path / dataname)
    has_ann_id = "ann_id" in lazy.collect_schema().names()

    # post_id: the real ann_id when the file has it, else cut it out of section_id
    post_id = c("ann_id").cast(pl.String) if has_ann_id else post_id_from_section_id(c("section_id"))
    title_col = par.star.get("title_col")
    if title_col:
        title = c(title_col).cast(pl.String)
    else:
        print("    - No title column on STAR, titles skipped")
        title = pl.lit(None, dtype=pl.String)

    columns = [post_id.alias("post_id"), title.alias("title"), c(text_var).alias("text")]
    if sectioned:
        columns.append(c("section_id"))

    return (
        lazy.select(columns)
        .filter(
            c("text").is_not_null()  # a few obs have missing text but non-missing heading and rubrik
        )
        .slice(0, par.settings.nobs)
        .collect()
    )


def rename_jobcenter_obs(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rename rows with id 'Virksomheden har valgt at rekruttere via jobcentret' to unique IDs.

    Args:
        df (pl.DataFrame): Input DataFrame with a 'post_id' column.

    Returns:
        pl.DataFrame: DataFrame with unique IDs for jobcenter posts.
    """
    # Add a unique suffix to each duplicate "jobcenter" id using cumcount
    df = df.with_columns(
        pl.when(c("post_id") == "Virksomheden har valgt at rekruttere via jobcentret")
        .then("jobcenter_" + (pl.cum_count("post_id").over("post_id") + 1).cast(pl.String))
        .otherwise(c("post_id"))
        .alias("post_id")
    )
    return df


def load_excel(file_path: Path, par: DictConfig, sheet_name: str = "Sheet1") -> pl.DataFrame:
    """
    Load data from an Excel file and return post_id, title and text.

    Args:
        file_path (Path): Path to the Excel file.
        par (DictConfig): Configuration parameters (settings.nobs limits the number of posts).
        sheet_name (str, optional): Name of the sheet to load. Defaults to "Sheet1".

    Returns:
        pl.DataFrame: DataFrame representing the rows in the Excel file.
    """

    df = (
        pl.read_excel(
            file_path,
            sheet_name=sheet_name,
            columns=["ID", "Title", "Text"],
            schema_overrides={"ID": pl.String, "Title": pl.String, "Text": pl.String},
        )
        .rename({"ID": "post_id", "Title": "title", "Text": "text"})
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
    Rename jobcenter posts and keep only Danish texts. HTML is cleaned later by split_posts.

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

    return df


def print_drop_summary(parts: pl.DataFrame) -> None:
    """Print how many parts were kept or dropped per reason, and 20 random dropped headings."""
    counts = parts.with_columns(c("drop_reason").fill_null("kept")).group_by("drop_reason").len().sort("drop_reason")
    for reason, n in counts.iter_rows():
        print(f"    - {reason}: {n:,}")
    headings = parts.filter(c("drop_reason") == "heading")
    for heading in headings.sample(min(20, headings.height), seed=1)["text"]:
        print(f"      dropped heading: {heading}")


def export_texts(texts: pl.DataFrame, output_file: Path) -> None:
    """
    Export the texts to a Parquet file.
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
    sentences_file = data_dir / "sentences.parquet"

    # Load parameters
    full_par = OmegaConf.load(params_path)
    par = full_par.prepare

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()
    print_params(full_par)

    # Load
    print("Loading data...")
    posts = load_data(file_path, par)
    num_texts_loaded = len(posts)

    # Clean
    print("Cleaning data...")
    posts = clean_data(posts)
    num_texts_used = len(posts)
    from_path = file_path
    if par.star.usestar == 1:
        from_path = Path("STAR data")
    print(f"    - Use {num_texts_used:,}/{num_texts_loaded:,} texts from {from_path}")

    # Split: one row per sentence (or paragraph), with a drop reason for headings and sensitive text
    unit = "sentence" if par.settings.split_sentences else "paragraph"
    if "section_id" in posts.columns:
        # STAR: a post has several sections, so section_id keeps the labels unique
        posts = posts.with_columns(c("section_id").alias("label_base"))
    parts = split_posts(posts, unit)
    print(f"    - Split {num_texts_used:,} texts into {parts.height:,} {unit}s")
    print_drop_summary(parts)

    # Keep the titles and the parts without a drop reason
    one_row_per_post = posts.unique("post_id", keep="first", maintain_order=True)  # STAR repeats a title per section
    texts = pl.concat([title_rows(one_row_per_post), parts.filter(c("drop_reason").is_null())]).drop("drop_reason")

    # Give every distinct text one sentence_id, so each distinct text is embedded only once
    texts, sentences = deduplicate(texts)
    share_duplicate = 1 - sentences.height / texts.height
    print(f"    - {sentences.height:,} unique texts out of {texts.height:,} rows ({share_duplicate:.1%} duplicates)")

    # Save
    export_texts(texts, texts_file)
    export_texts(sentences, sentences_file)
    print(f"Exported texts to {texts_file} and unique sentences to {sentences_file}")

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Save metrics
    with ((output_dir / "metrics") / "prepare.json").open("w") as f:
        json.dump({f"{Path(__file__).name}": f"{hours:.2f} hours"}, f, indent=4)
