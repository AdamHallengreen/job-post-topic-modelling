import time
from pathlib import Path
from typing import Any

import numpy as np
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
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
from job_post_topic_modelling.utils.data_io import (  # noqa: E402
    load_danish_stop_words,
    load_data,
    load_pretrained_embeddings,
)
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


class InvalidSeedTopicListError(Exception):
    """Exception raised when the seed topic list is not a list of lists as required by BERTopic."""

    def __init__(self, value):
        message = f"Invalid seed topic list: {value}. It must be a list of lists for BERTopic's seed_topic_list."
        super().__init__(message)


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
    print("Loading data...")
    documents = load_data(data_dir / "texts.parquet", text_col="text")
    embeddings = load_pretrained_embeddings(data_dir / "embeddings.npy")
    stop_words = load_danish_stop_words(data_dir / "stopwords-da.json")

    # Choose models
    embedding_model = get_embedding_model(embedding_model_name)
    dimensionality_reduction_model = get_dimensionality_reduction_model(par, embeddings=embeddings)
    clustering_model = get_clustering_model(par)
    vectorizer_model = CountVectorizer(stop_words=stop_words)

    # Handle seed topic input
    if par.seed_topics.seeds is not None:
        seed_topic_list = OmegaConf.to_container(par.seed_topics.seeds, resolve=True)
        if not isinstance(seed_topic_list, list) or not all(isinstance(x, list) for x in seed_topic_list):
            raise InvalidSeedTopicListError(seed_topic_list)
    else:
        seed_topic_list = None

    # Run BERTopic
    print("Training BERTopic model...")
    topic_model = BERTopic(
        # Modules
        embedding_model=embedding_model,
        umap_model=dimensionality_reduction_model,
        hdbscan_model=clustering_model,
        vectorizer_model=vectorizer_model,
        # Seeded topics
        seed_topic_list=seed_topic_list,
        # Topic distributions
        calculate_probabilities=par.settings.calculate_probabilities,
        # Hyperparameters
        top_n_words=par.settings.top_n_words,
        nr_topics="auto",
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
    print(f"Saved BERTopic model to {models_dir / 'bertopic_model'}")

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True, resume=True) as live:
        live.log_metric(f"{Path(__file__).name}", f"{hours:.2f} hours", plot=False)
