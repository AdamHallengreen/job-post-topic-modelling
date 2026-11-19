import json
import os
import pickle
import time
from pathlib import Path

# thread limits; set here instead of in dvc.yaml so the stage also runs under cmd.exe on Windows
for _name in (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "8")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from bertopic import BERTopic  # noqa: E402
from bertopic.cluster import BaseCluster  # noqa: E402
from bertopic.dimensionality import BaseDimensionalityReduction  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from polars import col as c  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from job_post_topic_modelling.utils.miscellaneous import try_inter  # noqa: E402

try_inter()
from job_post_topic_modelling.embedding_models import load_embedding_model  # noqa: E402
from job_post_topic_modelling.train import aggregate_predictions_to_ann_level  # noqa: E402
from job_post_topic_modelling.utils.data_io import (  # noqa: E402
    join_on_sentences,
    load_danish_stop_words,
    load_pretrained_embeddings,
)
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402
from job_post_topic_modelling.utils.miscellaneous import print_params  # noqa: E402


def add_wfh(texts, topics_wide):
    """Flag work-from-home mentions per occurrence (wfh_dummy) and per post (max over its occurrences)."""
    pattern = (
        r"\barbejd(e[rts]?|et)?(\s+\S+){0,2}\s+hjemme\s?fra\b"
        # 'arbjd hjemmefra' men fleksibelt iforhold til stavning og op til 2 ord imellem
    )

    df = texts.select(
        c.ann_id,
        c.label,
        wfh_dummy=pl.all_horizontal(
            pl.any_horizontal(
                c.text.str.to_lowercase().str.contains_any([
                    "hjemmearbejd",
                    "hjemmekontor",
                    "remote arbejde",
                    "work-from-home",
                    "Work from home",
                    "arbejdspladsen er hjemme",
                    "remote work",
                ]),
                # f'udføres {hjemmefra}',f'opgaver {hjemmefra}']
                c.text.str.to_lowercase().str.contains(pattern),
            ),
            ~c.text.str.contains_any(["væk hjemmefra", "langt hjemmefra"]),
        ),
    )

    texts = texts.join(df.select("label", "wfh_dummy"), on="label")

    df_wide = df.group_by("ann_id").agg(c.wfh_dummy.max())

    topics_wide = topics_wide.join(df_wide, on="ann_id")
    return texts, topics_wide


def topic_similarities(embeddings: np.ndarray, topic_embeddings: np.ndarray, ignore_limit: float | None) -> np.ndarray:
    """Cosine similarity between each sentence and each topic.

    Negative similarities are set to 0. Sentences whose best similarity is below `ignore_limit`
    get 0 for every topic.
    """
    similarities = cosine_similarity(embeddings, topic_embeddings)
    similarities[similarities < 0] = 0
    if ignore_limit is not None:
        ignore_index = np.max(similarities, axis=1) < ignore_limit
        print(f"Number of ignored sentences in this batch: {np.sum(ignore_index)}")
        similarities[ignore_index, :] = 0
    return similarities


def best_topics(similarities: np.ndarray, ignore_limit: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Most similar topic per sentence and its similarity.

    Sentences whose best similarity is below `ignore_limit` get topic -1 and probability 1.0.
    """
    topics = np.argmax(similarities, axis=1)
    probs = np.take_along_axis(similarities, topics[:, None], axis=1)[:, 0]
    if ignore_limit is not None:
        outliers = np.max(similarities, axis=1) < ignore_limit
        topics[outliers] = -1
        probs[outliers] = 1.0
    return topics, probs


def batch_ranges(n_sentences: int, shard_size: int):
    """Yield (start, end) pairs that cover the unique sentences in batches of `shard_size`."""
    for start_idx in range(0, n_sentences, shard_size):
        yield start_idx, min(start_idx + shard_size, n_sentences)


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

    # load: predictions are made once per unique sentence (sentence_id i = embedding row i)
    print("Loading data...")
    texts = pl.read_parquet(data_dir / "texts.parquet")
    sentences = pl.read_parquet(data_dir / "sentences.parquet")
    if par.settings.nobs is not None:
        sentences = sentences.filter(c.sentence_id < par.settings.nobs)
    documents = sentences["text"].to_list()
    sentence_ids = sentences["sentence_id"]

    embeddings = load_pretrained_embeddings(data_dir / "embeddings", nobs=par.settings.nobs)
    training_labels = pl.read_parquet(models_dir / "submodels/training_labels.parquet")["label"]

    print("Loading model")
    stop_words = load_danish_stop_words(data_dir / "stopwords-da.json")

    embedding_model = load_embedding_model(full_par.embed.model, full_par.embed.settings, device="cpu")

    topic_model = BERTopic.load(models_dir / "bertopic_model", embedding_model=embedding_model)
    # topic_model.seed_topics = seed_topic_list # check if needed

    if par.settings.clustering:
        print("Using clustering model for predictions...")
        with open(models_dir / r"submodels/dimensionality_reduction_model.pkl", "rb") as f:
            dimensionality_reduction_model = pickle.load(f)  # noqa: S301
        with open(models_dir / r"submodels/clustering_model.pkl", "rb") as f:
            clustering_model = pickle.load(f)  # noqa: S301

        if par_train.dimensionality_reduction.get("use_cuml", False):
            dimensionality_reduction_model.n_features_in_ = embeddings.shape[1]  # Monkey patch for cuml version
        topic_model.umap_model = dimensionality_reduction_model
        topic_model.hdbscan_model = clustering_model

    else:
        print("Predicting using cosine similarity (no clustering model)...")
        topic_model.hdbscan_model = BaseCluster()
        topic_model.umap_model = BaseDimensionalityReduction()

    n_sentences = len(documents)
    if (par.settings.batch_mode) and (par_train.settings.nobs is not None):
        shard_size = par_train.settings.nobs
        print("Predicting topics in batches...")
    else:
        print("Predicting topics for all unique sentences at once...")
        shard_size = n_sentences

    topic_embeddings = np.array(topic_model.topic_embeddings_[1:])  # Exclude topic -1
    ignore_limit = par.settings.ignore_limit
    if ignore_limit is not None:
        print(f"Ignoring sentences with max topic prob < {ignore_limit}...")

    if not par.settings.full_p_dist:
        print("Predicting most likely topics...")
        topics = np.array([], dtype=int)
        probs = np.array([], dtype=float)
        for start_idx, end_idx in batch_ranges(n_sentences, shard_size):
            print(f"Processing unique sentences {start_idx} to {end_idx}...")
            batch_embeddings = embeddings[start_idx:end_idx]
            if par.settings.clustering:
                batch_topics, batch_probs = topic_model.transform(
                    documents[start_idx:end_idx], embeddings=batch_embeddings
                )
            else:
                sim_matrix = topic_similarities(batch_embeddings, topic_embeddings, ignore_limit)
                batch_topics, batch_probs = best_topics(sim_matrix, ignore_limit)

            topics = np.concatenate([topics, batch_topics])
            probs = np.concatenate([probs, batch_probs])

        per_sentence = pl.DataFrame({
            "sentence_id": sentence_ids,
            "predicted_topic": topics,
            "topic_probability": probs,
        })
        texts = join_on_sentences(texts, per_sentence).with_columns(
            training_data=c.label.is_in(training_labels.implode()),
        )
        topics_wide = aggregate_predictions_to_ann_level(texts)

    else:
        print("Predicting full probability distributions...")
        if par.settings.clustering:
            raise ValueError("Full probability distributions are not compatible with clustering.")  # noqa: TRY003

        probs = np.vstack([
            topic_similarities(embeddings[start_idx:end_idx], topic_embeddings, ignore_limit)
            for start_idx, end_idx in batch_ranges(n_sentences, shard_size)
        ])

        per_sentence = pl.DataFrame(
            probs,
            schema=[f"topic_{i}" for i in range(probs.shape[1])],
        ).with_columns(sentence_id=sentence_ids)
        texts = join_on_sentences(texts, per_sentence).with_columns(
            training_data=c.label.is_in(training_labels.implode()),
        )

        topics_wide = texts.group_by("ann_id").agg(
            *[
                (pl.lit(1) - (pl.lit(1) - c(f"topic_{ti}")).product()).alias(f"p_topic_{ti}")
                for ti in range(probs.shape[1])
            ],
            pl.col("training_data").max(),
        )

    print("Adding working from home...")
    texts, topics_wide = add_wfh(texts, topics_wide)

    print("Saving predictions at sentence level...")
    # Save results
    texts.write_parquet(output_dir / "predicted_topics.parquet")

    print("Saving aggregated results...")
    topics_wide.write_parquet(output_dir / "predicted_topics_agg.parquet")

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Log metrics using
    with ((output_dir / "metrics") / "predict.json").open("w") as f:
        json.dump({f"{Path(__file__).name}": f"{hours:.2f} hours"}, f, indent=4)
