import json
import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.dimensionality import BaseDimensionalityReduction
from dvclive import Live
from omegaconf import OmegaConf
from polars import col as c

from job_post_topic_modelling.utils.miscellaneous import try_inter

try_inter()
from job_post_topic_modelling.train import aggregate_predictions_to_ann_level, get_embedding_model_cpu  # noqa: E402
from job_post_topic_modelling.utils.data_io import (  # noqa: E402
    load_danish_stop_words,
    load_pretrained_embeddings,
)
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402
from job_post_topic_modelling.utils.miscellaneous import print_params  # noqa: E402

if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"

    # Load parameters
    full_par = OmegaConf.load(params_path)
    par = full_par.predict
    par_train = full_par.train

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()
    print_params(full_par)

    # load
    print("Loading data...")
    texts = pl.read_parquet(data_dir / "texts.parquet")
    texts_train = texts.head(par_train.settings.nobs) if par_train.settings.nobs is not None else texts
    documents_train = texts_train["text"].to_list()

    if par.settings.nobs is not None:
        texts = texts.head(par.settings.nobs)
    documents = texts["text"].to_list()

    embeddings = load_pretrained_embeddings(data_dir / "embeddings", nobs=par.settings.nobs)

    print("Loading model")
    stop_words = load_danish_stop_words(data_dir / "stopwords-da.json")
    embedding_model_name = full_par.embed.model.embedding_model

    embedding_model = get_embedding_model_cpu(embedding_model_name)

    topic_model = BERTopic.load(models_dir / "bertopic_model", embedding_model=embedding_model)
    # topic_model.seed_topics = seed_topic_list # check if needed

    if par.settings.clustering:
        print("Using clustering model for predictions...")
        with open(models_dir / r"submodels/dimensionality_reduction_model.pkl", "rb") as f:
            dimensionality_reduction_model = pickle.load(f)  # noqa: S301
        with open(models_dir / r"submodels/clustering_model.pkl", "rb") as f:
            clustering_model = pickle.load(f)  # noqa: S301

        if par_train.dimensionality_reduction.use_cuml:
            dimensionality_reduction_model.n_features_in_ = embeddings.shape[1]  # Monkey patch for cuml version
        topic_model.umap_model = dimensionality_reduction_model
        topic_model.hdbscan_model = clustering_model

    else:
        print("Predicting using cosine similarity (no clustering model)...")
        topic_model.hdbscan_model = BaseCluster()
        topic_model.umap_model = BaseDimensionalityReduction()

    if (par.settings.batch_mode) and (par_train.settings.nobs is not None):
        shard_size = par_train.settings.nobs
        print("Predicting topics in batches...")
    else:
        print("Predicting topics for all documents at once...")
        shard_size = len(documents)

    topics = np.array([], dtype=int)
    probs = np.array([], dtype=float)
    stop = par.settings.nobs if par.settings.nobs is not None else len(documents)
    stop = min(stop, len(documents))

    for start_idx in range(0, stop, shard_size):
        end_idx = min(start_idx + shard_size, stop)
        print(f"Processing documents {start_idx} to {end_idx}...")
        batch_docs = documents[start_idx:end_idx]
        batch_embeddings = embeddings[start_idx:end_idx]
        batch_topics, batch_probs = topic_model.transform(batch_docs, embeddings=batch_embeddings)

        topics = np.concatenate([topics, batch_topics])
        probs = np.concatenate([probs, batch_probs])

    texts = (
        texts.with_row_index("row_nr")
        .with_columns(
            pl.Series("predicted_topic", topics),
            pl.Series("topic_probability", probs),
            ann_id=c.label.str.extract(r"^(\d+)_s", 1),
            training_data=(pl.col("row_nr") < par_train.settings.nobs),
        )
        .drop("row_nr")
    )

    print("Saving predictions at sentence level...")

    # Save results
    texts.write_parquet(output_dir / "predicted_topics.parquet")

    topics_wide = aggregate_predictions_to_ann_level(texts)

    print("Saving aggregated results...")
    topics_wide.write_parquet(output_dir / "predicted_topics_agg.parquet")

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Log metrics using
    with ((output_dir / "metrics") / "predict.json").open("w") as f:
        json.dump({f"{Path(__file__).name}": f"{hours:.2f} hours"}, f, indent=4)
