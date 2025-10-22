import json
from pathlib import Path

import numpy as np
import polars as pl


def load_data(filepath: Path | str, text_col: str = "text") -> list[str]:
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


def load_pretrained_embeddings(filepath: Path | str, obs_stop=None):
    """
    Load all embeddings from shards and concatenate them.
    """
    filepath = Path(filepath)

    all_embeddings = []
    shard_files = sorted(filepath.glob("embeddings_shard_*.npy"))
    shards = len(shard_files)
    shards_loaded = 0
    total_obs = 0

    for shard_file in shard_files:
        shard_embeddings = np.load(shard_file)
        all_embeddings.append(shard_embeddings)
        total_obs += shard_embeddings.shape[0]
        shards_loaded += 1
        if obs_stop is not None and total_obs >= obs_stop:
            break

    embeddings = np.vstack(all_embeddings)

    print(f"Loaded {embeddings.shape[0]} embeddings from {shards_loaded} out of {shards} shards.")

    return embeddings


def load_danish_stop_words(filepath: str | Path) -> list[str]:
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
