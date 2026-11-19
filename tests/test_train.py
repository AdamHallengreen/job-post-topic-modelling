import numpy as np
import polars as pl
from omegaconf import OmegaConf

from job_post_topic_modelling.train import (
    expand_to_occurrences,
    get_clustering_model,
    resolve_cluster_sizes,
    sample_training_rows,
)
from job_post_topic_modelling.utils.data_io import join_on_sentences, load_pretrained_embeddings


def test_load_selected_rows_across_shards(tmp_path):
    np.save(tmp_path / "embeddings_shard_1.npy", np.arange(6).reshape(3, 2))
    np.save(tmp_path / "embeddings_shard_2.npy", np.arange(6, 10).reshape(2, 2))
    out = load_pretrained_embeddings(tmp_path, rows=np.array([1, 3]))
    np.testing.assert_array_equal(out, [[2, 3], [6, 7]])


def test_join_on_sentences_gives_every_copy_the_same_topic():
    texts = pl.DataFrame({"post_id": ["a", "b", "a"], "label": ["a_00", "b_00", "a_01"], "sentence_id": [0, 0, 1]})
    per_sentence = pl.DataFrame({"sentence_id": [1, 0], "predicted_topic": [7, 3]})
    out = join_on_sentences(texts, per_sentence)
    assert out["label"].to_list() == ["a_00", "b_00", "a_01"]
    assert out["predicted_topic"].to_list() == [3, 3, 7]
    assert out["ann_id"].to_list() == ["a", "b", "a"]


def test_resolve_cluster_sizes_scales_with_data():
    assert resolve_cluster_sizes(8_000_000, 0.0005, 0.1) == (4000, 400)
    assert resolve_cluster_sizes(280_000, 0.0005, 0.1) == (140, 14)
    assert resolve_cluster_sizes(1_000, 0.0005, 0.1) == (10, 1)  # floor


def test_sample_training_rows():
    rows = sample_training_rows(100, nobs=10, seed=42)
    assert len(rows) == 10 and len(set(rows)) == 10 and list(rows) == sorted(rows)
    np.testing.assert_array_equal(rows, sample_training_rows(100, nobs=10, seed=42))
    np.testing.assert_array_equal(sample_training_rows(5, nobs=None, seed=42), np.arange(5))
    np.testing.assert_array_equal(sample_training_rows(5, nobs=50, seed=42), np.arange(5))


def test_expand_to_occurrences_repeats_each_copy():
    unique_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = expand_to_occurrences(unique_embeddings, unique_ids=np.array([3, 7]), occurrence_ids=np.array([7, 3, 7, 7]))
    np.testing.assert_array_equal(out, [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])


def test_get_clustering_model_resolves_relative_sizes_only_when_given():
    relative = OmegaConf.create({
        "clustering": {"model": "HDBSCAN", "min_cluster_size_share": 0.0005, "min_samples_ratio": 0.1}
    })
    model = get_clustering_model(relative, n_train=280_000, seed=42, n_jobs=3)
    assert (model.min_cluster_size, model.min_samples, model.core_dist_n_jobs) == (140, 14, 3)

    blank = OmegaConf.create({"clustering": {"model": "HDBSCAN"}})
    model = get_clustering_model(blank, n_train=280_000, seed=42, n_jobs=3)
    assert model.min_cluster_size == 5 and model.min_samples is None  # library defaults
