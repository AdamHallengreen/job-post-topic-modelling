
from sentence_transformers import SentenceTransformer

import json
import pathlib
import time
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired


from hdbscan import HDBSCAN
import polars as pl
from bertopic import BERTopic
from dvclive import Live
from omegaconf import DictConfig, ListConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from umap import UMAP

from job_post_topic_modelling.utils.interactive import try_inter

try_inter()
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402



class UnsupportedAnchorTypeError(Exception):
    """Exception raised when an unsupported anchor type is encountered."""

    def __init__(self, anchor: Any):
        message = f"Unsupported anchor type: {type(anchor).__name__}.Anchor must be a string or a list of strings."
        super().__init__(message)

def load_data(filepath: str = None, text_col: str = "text") -> list[str]:
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

def load_danish_stopwords(filepath: str = None) -> list[str]:
    """
    Load Danish stop words from a JSON file.
    Args:
        filepath (str, optional): Path to the stopwords file.
    Returns:
        list[str]: List of Danish stop words.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        stopwords = json.load(f)
    return stopwords

def get_sentence_transformer():
    sentence_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
    return sentence_model

def get_dimensionality_reduction_model():
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
    return umap_model

def get_clustering_model():
    #cluster_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    cluster_model = KMeans(n_clusters=10)
    return cluster_model

def get_vectorizer(stop_words):
    vectorizer_model = CountVectorizer(stop_words=stop_words, ngram_range=(1, 3))
    return vectorizer_model

def get_ctfidf_model():
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    return ctfidf_model

def get_representation_model():
    representation_model = KeyBERTInspired()
    return representation_model

if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"

    # Load parameters
    par = OmegaConf.load(params_path).train

    start = time.time()

    # Load data as list of texts for BERTopic
    documents = load_data(data_dir / "texts.parquet", text_col="text")

    # Get sentence transformer model
    print("Loading sentence transformer model...")
    sentence_model = get_sentence_transformer()

    # Get dimensionality reduction model
    umap_model = get_dimensionality_reduction_model()

    # Get clustering model
    cluster_model = get_clustering_model()

    # Get vectorizer model
    stop_words = load_danish_stopwords(data_dir / "stopwords-da.json")
    vectorizer_model = get_vectorizer(stop_words=stop_words)

    # Get c-TF-IDF model
    ctfidf_model = get_ctfidf_model()

    # Get representation model
    representation_model = get_representation_model()

    print("Fitting BERTopic model...")
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model
    )
    topics, probs = topic_model.fit_transform(documents)

    topic_model.get_topic_info()

    # Save topics to output directory
    topics_out_path = output_dir / "bertopic_topics.parquet"
    topics_df = pl.DataFrame({
        "document": documents,
        "topic": topics,
        "probability": probs
    })
    topics_df.write_parquet(str(topics_out_path))
    print(f"Saved BERTopic topics to {topics_out_path}")

    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished train.py in {hours:.2f} hours")
    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True, resume=True) as live:
        # Log metrics
        live.log_metric("train.py", f"{hours:.2f} hours", plot=False)