import os
import time
from pathlib import Path

import polars as pl
from dvclive import Live
from lingua import LanguageDetectorBuilder
from omegaconf import DictConfig, OmegaConf

from job_post_topic_modelling.utils.interactive import try_inter

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
    if par.star.usestar > 0:
        return load_star_data(par.star.usestar)

    # check if the file exists
    if not file_path.exists():
        raise FileNotFoundError(FileNotFoundErrorMessage(file_path))
    # check if the file is an Excel file
    if file_path.suffix in [".xlsx", ".xls"]:
        df = load_excel(file_path)
    else:
        raise UnsupportedFileTypeError(file_path.suffix)
    return df


def load_star_data(usestar) -> pl.DataFrame:
    """
    Loads the jobpost data from star data on the server
    """
    # get username
    username = os.popen("whoami").read().strip()  # noqa: S607 S605

    folder_path = Path(f"/home/{username}@PROD.SITAD.DK/code/jobads/src/dgp/textdata/output")

    if usestar == 1:
        dataname = "jobads_clean.parquet"
        id_var = "ann_id"
        text_var = "annonce_tekst"
    elif usestar == 2:
        dataname = "jobads_sections_clean.parquet"
        id_var = "section_id"
        text_var = "section_text"

    df = (
        pl.read_parquet(folder_path / dataname)
        .select(pl.col(id_var).alias("id"), pl.col(text_var).alias("text"))
        .filter(
            pl.col("text").is_not_null()  # a few obs have missing text but non-missing heading and rubrik
        )
    )
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
        pl.when(pl.col("id") == "Virksomheden har valgt at rekruttere via jobcentret")
        .then("jobcenter_" + (pl.cum_count("id").over("id") + 1).cast(pl.String))
        .otherwise(pl.col("id"))
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
    df = pl.read_excel(
        file_path, sheet_name=sheet_name, columns=["ID", "Text"], schema_overrides={"ID": pl.String, "Text": pl.String}
    ).rename({"ID": "id", "Text": "text"})
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
    texts = df.select(pl.col("text")).to_series().to_list()

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
    df = df.filter(pl.col("language") == "DAN")

    return df


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
    par = OmegaConf.load(params_path).prepare

    start = time.time()
    print("Starting prepare.py")
    # Process the data
    texts = load_data(file_path, par)[: par.settings.nobs]
    texts = detect_language(texts)
    texts = clean_data(texts)
    print(f"Loaded {len(texts)} texts from {file_path}")

    export_texts(texts, texts_file)
    print(f"Texts exported to {texts_file}")

    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished prepare.py in {hours:.2f} hours")
    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True) as live:
        # Log metrics
        live.log_metric("prepare.py", f"{hours:.2f} hours", plot=False)
