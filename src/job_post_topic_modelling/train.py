import json
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from hdbscan import HDBSCAN
from omegaconf import OmegaConf
from polars import col as c
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from job_post_topic_modelling.embedding_models import load_embedding_model
from job_post_topic_modelling.utils.miscellaneous import print_params, try_inter

try_inter()
from job_post_topic_modelling.utils.data_io import (  # noqa: E402
    load_danish_stop_words,
    load_pretrained_embeddings,
)
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402


class UnknownModelError(Exception):
    """Exception raised when an unknown model is encountered."""

    def __init__(self, model_name: str):
        super().__init__(f"Unknown model: {model_name}")


class InvalidSeedTopicListError(Exception):
    """Exception raised when the seed topic list is not a list of lists as required by BERTopic."""

    def __init__(self, value):
        message = f"Invalid seed topic list: {value}. It must be a list of lists for BERTopic's seed_topic_list."
        super().__init__(message)


def resolve_cluster_sizes(
    n_train: int, min_cluster_size_share: float, min_samples_ratio: float, floor: int = 10
) -> tuple[int, int]:
    """Turn relative HDBSCAN settings into absolute (min_cluster_size, min_samples) for n_train occurrences."""
    min_cluster_size = max(floor, round(min_cluster_size_share * n_train))
    min_samples = max(1, round(min_samples_ratio * min_cluster_size))
    return min_cluster_size, min_samples


def sample_training_rows(n_rows: int, nobs: int | None, seed: int) -> np.ndarray:
    """Pick a random sample of occurrence rows (sorted, no repeats). All rows if nobs is None or >= n_rows."""
    if nobs is None or nobs >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, nobs, replace=False))


def expand_to_occurrences(
    unique_embeddings: np.ndarray, unique_ids: np.ndarray, occurrence_ids: np.ndarray
) -> np.ndarray:
    """Return one embedding row per entry in occurrence_ids.

    unique_embeddings[i] is the embedding of sentence unique_ids[i]; unique_ids must be sorted.
    A sentence that occurs N times in occurrence_ids is copied N times, so UMAP and HDBSCAN see it N times.
    """
    # unique_ids is sorted, so searchsorted gives, for each occurrence, the position of its sentence in unique_ids
    # (which is also its row in unique_embeddings).
    positions = np.searchsorted(unique_ids, occurrence_ids)
    return unique_embeddings[positions]


def get_dimensionality_reduction_model(par: OmegaConf, seed: int, n_jobs: int):
    args = {k: v for k, v in par.dimensionality_reduction.items() if (k not in ["model"])}

    if par.dimensionality_reduction.model == "UMAP":
        # A fixed random_state makes UMAP single-threaded; umap_seed=false trades reproducibility for speed.
        random_state = seed if par.settings.umap_seed else None
        dimensionality_reduction_model = UMAP(init="pca", random_state=random_state, n_jobs=n_jobs, **args)
    elif par.dimensionality_reduction.model == "PCA":
        dimensionality_reduction_model = PCA(**args)
    elif par.dimensionality_reduction.model == "empty":
        dimensionality_reduction_model = BaseDimensionalityReduction()
    else:
        raise UnknownModelError(par.dimensionality_reduction.model)
    return dimensionality_reduction_model


def get_clustering_model(par: OmegaConf, n_train: int, seed: int, n_jobs: int):
    args = {k: v for k, v in par.clustering.items() if k not in ["model"]}
    if par.clustering.model == "HDBSCAN":
        if "min_cluster_size_share" in args:
            # Relative settings: turn them into absolute sizes for this training set
            min_cluster_size, min_samples = resolve_cluster_sizes(
                n_train, args.pop("min_cluster_size_share"), args.pop("min_samples_ratio")
            )
            args.update(min_cluster_size=min_cluster_size, min_samples=min_samples)
        # Otherwise (e.g. HDBSCAN_blank) the args pass through unchanged: library defaults
        clustering_model = HDBSCAN(core_dist_n_jobs=n_jobs, **args)
    elif par.clustering.model == "KMeans":
        clustering_model = KMeans(random_state=seed, **args)
    else:
        raise UnknownModelError(par.clustering.model)
    return clustering_model


def load_seed_topic_list(par: OmegaConf):
    """Read the optional seed topics (a list of lists of words) for BERTopic."""
    if par.seed_topics.seeds is None:
        return None
    seed_topic_list = OmegaConf.to_container(par.seed_topics.seeds, resolve=True)
    if not isinstance(seed_topic_list, list) or not all(isinstance(x, list) for x in seed_topic_list):
        raise InvalidSeedTopicListError(seed_topic_list)
    return seed_topic_list


def aggregate_predictions_to_ann_level(df):
    print("Aggregate to ann_id/job add level")
    topics_agg = df.group_by(
        "ann_id",
        "predicted_topic",
    ).agg(
        (pl.lit(1) - (pl.lit(1) - c("topic_probability")).product()).alias("p_topic"),
        (pl.col("topic_probability").max() > 0).alias("d_topic"),
        pl.col(
            "training_data"
        ).max(),  # during loading of embeddings, half of one add might have been used for training
    )
    print("Make into wide format")
    topics_wide = (
        topics_agg.sort("predicted_topic").pivot(
            values=["p_topic", "d_topic"],
            index=["ann_id", "training_data"],
            on="predicted_topic",
            aggregate_function=None,
        )
    ).with_columns(
        cs.starts_with("p_topic").fill_null(0.0),
        cs.starts_with("d_topic").fill_null(False),
    )
    return topics_wide


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"

    # Load parameters
    full_par = OmegaConf.load(params_path)
    par = full_par.train

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()
    print_params(full_par)

    # 1. Load every sentence occurrence (one row per sentence in each post)
    print("Loading data...")
    texts = pl.read_parquet(data_dir / "texts.parquet")
    stop_words = load_danish_stop_words(data_dir / "stopwords-da.json")

    # 2. Sample training occurrences. A sentence that occurs N times can be drawn up to N times.
    train_texts = texts[sample_training_rows(texts.height, par.settings.nobs, par.settings.seed)]

    # 3. Load the embeddings of only the unique sentences in the sample (each sentence is embedded once)
    occurrence_ids = train_texts["sentence_id"].to_numpy()
    unique_ids = np.unique(occurrence_ids)
    unique_embeddings = load_pretrained_embeddings(data_dir / "embeddings", rows=unique_ids)

    # 4. Expand back to one embedding per occurrence, so UMAP and HDBSCAN see repeated sentences repeatedly
    embeddings = expand_to_occurrences(unique_embeddings, unique_ids, occurrence_ids)
    documents = train_texts["text"].to_list()
    print(f"{len(documents)} occurrences, {len(unique_ids)} unique sentences")

    # Choose models
    embedding_model = load_embedding_model(full_par.embed.model, full_par.embed.settings, device="cpu")
    dimensionality_reduction_model = get_dimensionality_reduction_model(par, par.settings.seed, par.settings.n_jobs)
    clustering_model = get_clustering_model(par, len(documents), par.settings.seed, par.settings.n_jobs)
    seed_topic_list = load_seed_topic_list(par)
    vectorizer_model = CountVectorizer(stop_words=stop_words)
    # The fitted HDBSCAN knows the resolved sizes (KMeans has none)
    min_cluster_size = getattr(clustering_model, "min_cluster_size", None)
    min_samples = getattr(clustering_model, "min_samples", None)
    print(f"min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    # 5. Run BERTopic. Topics and probabilities come out per occurrence.
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
        nr_topics=par.settings.nr_topics,
        verbose=par.settings.verbose,
    )
    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)

    # 6. Optionally move outlier occurrences (topic -1) into the nearest topic, then refresh topic words
    if par.settings.reduce_outliers is not None:
        topics = topic_model.reduce_outliers(
            documents,
            topics,
            strategy=par.settings.reduce_outliers,
            embeddings=embeddings,
            threshold=par.settings.reduce_outliers_threshold,
        )
        topic_model.update_topics(
            documents, topics=topics, vectorizer_model=vectorizer_model, top_n_words=par.settings.top_n_words
        )

    # 7. Save the model, the submodels and the labels of the training occurrences
    print(f"Saving BERTopic model to {models_dir / 'bertopic_model'}")
    topic_model.save(
        models_dir / "bertopic_model",
        serialization="safetensors",
        save_ctfidf=False,  # True, # There is some error here for TRUE
        save_embedding_model=False,
    )
    print("Saving sub models ... ")
    with (models_dir / "submodels/dimensionality_reduction_model.pkl").open("wb") as f:
        pickle.dump(topic_model.umap_model, f)
    with (models_dir / "submodels/clustering_model.pkl").open("wb") as f:
        pickle.dump(topic_model.hdbscan_model, f)
    train_texts.select("label").write_parquet(models_dir / "submodels/training_labels.parquet")

    # 8. Predictions per training occurrence (no join needed: topics are already per occurrence)
    print("Saving predictions on training data...")
    predictions = train_texts.with_columns(
        predicted_topic=pl.Series(topics),
        topic_probability=pl.Series(probs),  # 1-D (one float per occurrence) unless calculate_probabilities is true
        ann_id=c.post_id,
        training_data=pl.lit(True),
    )

    # 9. Save results
    predictions.write_parquet(output_dir / "predicted_topics_train.parquet")
    topics_wide = aggregate_predictions_to_ann_level(predictions)
    print("Saving aggregated results...")
    topics_wide.write_parquet(output_dir / "predicted_topics_agg_train.parquet")

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # 10. Log metrics
    with ((output_dir / "metrics") / "train.json").open("w") as f:
        json.dump(
            {
                f"{Path(__file__).name}": f"{hours:.2f} hours",
                "n_train_occurrences": len(documents),
                "n_train_unique_sentences": len(unique_ids),
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
            },
            f,
            indent=4,
        )
