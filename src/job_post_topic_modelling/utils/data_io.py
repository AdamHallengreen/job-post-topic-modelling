import json
from pathlib import Path

import numpy as np
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


def load_pretrained_embeddings(filepath: Path):
    """
    Load precomputed embeddings from output/embeddings.npy if available, otherwise return the embedding model.
    """
    if filepath.exists():
        embeddings = np.load(filepath)
        return embeddings


def load_danish_stop_words(filepath: str) -> list[str]:
    """
    Load Danish stop words from a JSON file.
    Args:
        filepath (str, optional): Path to the _rds file.
    Returns:
        list[str]: List of Danish stop words.
    """
    with open(filepath, encoding="utf-8") as f:
        stop_words = json.load(f)
    return stop_words
