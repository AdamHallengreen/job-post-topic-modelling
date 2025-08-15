import json
import time
from pathlib import Path
from typing import Any

import polars as pl
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from dvclive import Live
from hdbscan import HDBSCAN
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from job_post_topic_modelling.utils.interactive import try_inter

try_inter()
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402


class UnsupportedAnchorTypeError(Exception):
    """Exception raised when an unsupported anchor type is encountered."""

    def __init__(self, anchor: Any):
        message = f"Unsupported anchor type: {type(anchor).__name__}.Anchor must be a string or a list of strings."
        super().__init__(message)


class UnknownModelError(Exception):
    """Exception raised when an unknown model is encountered."""

    def __init__(self, model_name: str):
        super().__init__(f"Unknown model: {model_name}")


def load_data(filepath: str, text_col: str = "text") -> list[str]:
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


def load_danish_stopwords(filepath: str) -> list[str]:
    """
    Load Danish stop words from a JSON file.
    Args:
        filepath (str, optional): Path to the stopwords file.
    Returns:
        list[str]: List of Danish stop words.
    """
    with open(filepath, encoding="utf-8") as f:
        stopwords = json.load(f)
    return stopwords


def get_embedding_model(par: OmegaConf):
    sentence_model = SentenceTransformer(par.embedding.embedding_model)
    return sentence_model


def get_dimensionality_reduction_model(par: OmegaConf):
    if par.dimensionality_reduction.model == "UMAP":
        dimensionality_reduction_model = UMAP(
            n_neighbors=par.dimensionality_reduction.n_neighbors,
            n_components=par.dimensionality_reduction.n_components,
            min_dist=par.dimensionality_reduction.min_dist,
            metric=par.dimensionality_reduction.metric,
            random_state=par.dimensionality_reduction.random_state,
        )
    elif par.dimensionality_reduction.model == "PCA":
        dimensionality_reduction_model = PCA(
            n_components=par.dimensionality_reduction.n_components,
        )
    elif par.dimensionality_reduction.model == "empty":
        dimensionality_reduction_model = BaseDimensionalityReduction()
    else:
        raise UnknownModelError(par.dimensionality_reduction.model)
    return dimensionality_reduction_model


def get_clustering_model(par: OmegaConf):
    if par.clustering.model == "HDBSCAN":
        clustering_model = HDBSCAN(
            min_cluster_size=par.clustering.min_cluster_size,
            metric=par.clustering.metric,
            cluster_selection_method=par.clustering.cluster_selection_method,
            prediction_data=par.clustering.prediction_data,
        )
    elif par.clustering.model == "KMeans":
        clustering_model = KMeans(n_clusters=par.clustering.n_clusters)
    else:
        raise UnknownModelError(par.clustering.model)
    return clustering_model


def get_vectorizer(par: OmegaConf, stop_words=None):
    if par.vectorizer.model == "CountVectorizer":
        vectorizer_model = CountVectorizer(
            stop_words=stop_words,
            ngram_range=(par.vectorizer.ngram_range[0], par.vectorizer.ngram_range[1]),
            min_df=par.vectorizer.min_df,
            max_features=par.vectorizer.max_features,
        )
    else:
        raise UnknownModelError(par.vectorizer.model)
    return vectorizer_model


def get_cTFIDF_model(par: OmegaConf):
    if par.c_TF_IDF.model == "c_TF_IDF":
        ctfidf_model = ClassTfidfTransformer(
            bm25_weighting=par.c_TF_IDF.bm25_weighting, reduce_frequent_words=par.c_TF_IDF.reduce_frequent_words
        )
    else:
        raise UnknownModelError(par.c_TF_IDF.model)
    return ctfidf_model


def get_representation_model(par: OmegaConf):
    if par.representation.model == "KeyBERTInspired":
        representation_model = KeyBERTInspired()
    else:
        raise UnknownModelError(par.representation.model)
    return representation_model


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"

    # Load parameters
    par = OmegaConf.load(params_path).predict

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()

    # Load
    documents = load_data(data_dir / "texts.parquet", text_col="text")
    stop_words = load_danish_stopwords(data_dir / "stopwords-da.json")

    # Choose models
    embedding_model = get_embedding_model(par)
    dimensionality_reduction_model = get_dimensionality_reduction_model(par)
    clustering_model = get_clustering_model(par)
    vectorizer_model = get_vectorizer(par, stop_words=stop_words)
    ctfidf_model = get_cTFIDF_model(par)
    representation_model = get_representation_model(par)

    # Run BERTopic
    topic_model = BERTopic(
        # Modules
        embedding_model=embedding_model,
        umap_model=dimensionality_reduction_model,
        hdbscan_model=clustering_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        # Hyperparameters
        top_n_words=par.top_n_words,
        verbose=par.verbose,
    )
    topics, probs = topic_model.fit_transform(documents)

    # Save model
    topic_model.save(
        models_dir / "bertopic_model",
        serialization="safetensors",
        save_ctfidf=False,  # True, # There is some error here for TRUE
        save_embedding_model=par.embedding.embedding_model,
    )

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True, resume=True) as live:
        live.log_metric(f"{Path(__file__).name}", f"{hours:.2f} hours", plot=False)
