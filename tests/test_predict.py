import numpy as np
import polars as pl

from job_post_topic_modelling.predict import add_wfh, best_topics, topic_similarities


def test_topic_similarities_clip_and_ignore():
    emb = np.array([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0]])
    topics = np.array([[1.0, 0.0], [0.0, 1.0]])
    sims = topic_similarities(emb, topics, ignore_limit=0.5)
    np.testing.assert_allclose(sims, [[1.0, 0.0], [0.6, 0.8], [0.0, 0.0]])


def test_topic_similarities_without_ignore_limit_keeps_low_rows():
    emb = np.array([[0.6, 0.8]])
    topics = np.array([[1.0, 0.0], [0.0, 1.0]])
    sims = topic_similarities(emb, topics, ignore_limit=None)
    np.testing.assert_allclose(sims, [[0.6, 0.8]])


def test_best_topics_marks_low_similarity_as_outlier():
    sims = np.array([[0.9, 0.1], [0.2, 0.3]])
    topics, probs = best_topics(sims, ignore_limit=0.5)
    assert topics.tolist() == [0, -1]
    np.testing.assert_allclose(probs, [0.9, 1.0])


def test_add_wfh_uses_ann_id_column():
    texts = pl.DataFrame({
        "ann_id": [1, 1, 2],
        "label": ["1_s1", "1_s2", "2_s1"],
        "text": ["Du kan have hjemmekontor", "Vi er et hold", "Kontor i Aarhus"],
    })
    topics_wide = pl.DataFrame({"ann_id": [1, 2]})
    texts, topics_wide = add_wfh(texts, topics_wide)
    assert texts["wfh_dummy"].to_list() == [True, False, False]
    assert topics_wide.sort("ann_id")["wfh_dummy"].to_list() == [True, False]
