import json
import time
from pathlib import Path
from typing import Any

import numpy as np
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
from job_post_topic_modelling.utils.data_io import load_data, load_pretrained_embeddings  # noqa: E402
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


def rescale(x, inplace=False):
    """Rescale an embedding so optimization will not have convergence issues."""
    if not inplace:
        x = np.array(x, copy=True)

    x /= np.std(x[:, 0]) * 10000

    return x


def get_embedding_model(embedding_model_name: str):
    sentence_model = SentenceTransformer(embedding_model_name)
    return sentence_model


def get_dimensionality_reduction_model(par: OmegaConf, embeddings=None):
    args = {k: v for k, v in par.dimensionality_reduction.items() if k != "model"}
    if par.dimensionality_reduction.model == "UMAP":
        # Initialize and rescale PCA embeddings
        pca_embeddings = rescale(PCA(n_components=5).fit_transform(embeddings))
        # Start UMAP from PCA embeddings
        dimensionality_reduction_model = UMAP(init=pca_embeddings, **args)
    elif par.dimensionality_reduction.model == "PCA":
        dimensionality_reduction_model = PCA(**args)
    elif par.dimensionality_reduction.model == "empty":
        dimensionality_reduction_model = BaseDimensionalityReduction()
    else:
        raise UnknownModelError(par.dimensionality_reduction.model)
    return dimensionality_reduction_model


def get_clustering_model(par: OmegaConf):
    args = {k: v for k, v in par.clustering.items() if k != "model"}
    if par.clustering.model == "HDBSCAN":
        clustering_model = HDBSCAN(**args)
    elif par.clustering.model == "KMeans":
        clustering_model = KMeans(**args)
    else:
        raise UnknownModelError(par.clustering.model)
    return clustering_model


def get_vectorizer(par: OmegaConf, stop_words=None):
    args = {k: v for k, v in par.vectorizer.items() if k != "model"}
    if "ngram_range" in args:
        args["ngram_range"] = tuple(args["ngram_range"])
    if par.vectorizer.model == "CountVectorizer":
        vectorizer_model = CountVectorizer(stop_words=stop_words, **args)
    else:
        raise UnknownModelError(par.vectorizer.model)
    return vectorizer_model


def get_cTFIDF_model(par: OmegaConf):
    args = {k: v for k, v in par.c_TF_IDF.items() if k != "model"}
    if par.c_TF_IDF.model == "c_TF_IDF":
        ctfidf_model = ClassTfidfTransformer(**args)
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
    par = OmegaConf.load(params_path).train
    embedding_model_name = OmegaConf.load(params_path).embed.model.embedding_model

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()

    # Load
    documents = load_data(data_dir / "texts.parquet", text_col="text")
    embeddings = load_pretrained_embeddings(data_dir / "embeddings.npy")
    stop_words = load_danish_stop_words(data_dir / "stopwords-da.json")

    # Choose models
    embedding_model = get_embedding_model(embedding_model_name)
    dimensionality_reduction_model = get_dimensionality_reduction_model(par, embeddings=embeddings)
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
        top_n_words=par.settings.top_n_words,
        verbose=par.settings.verbose,
    )
    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

    # Save model
    topic_model.save(
        models_dir / "bertopic_model",
        serialization="safetensors",
        save_ctfidf=False,  # True, # There is some error here for TRUE
        save_embedding_model=embedding_model_name,
    )

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True, resume=True) as live:
        live.log_metric(f"{Path(__file__).name}", f"{hours:.2f} hours", plot=False)
