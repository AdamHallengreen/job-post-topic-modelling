import numpy as np
import polars as pl

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
