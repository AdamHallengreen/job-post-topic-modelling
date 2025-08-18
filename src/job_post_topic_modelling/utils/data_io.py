from pathlib import Path

import polars as pl


def load_data(filepath: Path, text_col: str = "text") -> list[str]:
    """
    Load the texts.parquet file and return a list of texts for BERTopic.
    Args:
        filepath (str, optional): Path to the texts.parquet file.
        text_col (str): Name of the column containing text data.
    Returns:
        list[str]: List of text documents.
    """
    df = pl.read_parquet(filepath)
    if text_col in df.columns:
        return df[text_col].to_list()
    else:
        # Fallback: use the first column
        return df[df.columns[0]].to_list()
