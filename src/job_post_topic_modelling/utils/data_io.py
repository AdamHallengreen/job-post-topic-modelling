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


def load_pretrained_embeddings(filepath: Path | str, nobs=None):
    """
    Load all embeddings from shards and concatenate them.
    """
    filepath = Path(filepath)

    all_embeddings = []
    shard_files = sorted(filepath.glob("embeddings_shard_*.npy")) # sorted sorts 1 10 11 ... not 1 2 3
    shards = len(shard_files)
    shards_loaded = 0
    total_obs = 0

    for shard_i in range(1,shards+1):
        shard_file = filepath / f"embeddings_shard_{shard_i}.npy"
        shard_embeddings = np.load(shard_file)
        all_embeddings.append(shard_embeddings)
        total_obs += shard_embeddings.shape[0]
        shards_loaded += 1
        if nobs is not None and total_obs >= nobs:
            print(f"Stopped loading at {total_obs} observations as requested. Will trim to {nobs}.")
            break

    embeddings = np.vstack(all_embeddings)[:nobs] if nobs is not None else np.vstack(all_embeddings)

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
