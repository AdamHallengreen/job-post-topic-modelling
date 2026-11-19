import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from job_post_topic_modelling.embedding_models import (
    encode,
    resolve_dtype,
    resolve_model_path,
    truncate_and_normalize,
)


def test_truncate_then_normalize():
    emb = np.array([[3.0, 4.0, 12.0], [0.0, 0.0, 5.0]])
    out = truncate_and_normalize(emb, truncate_dim=2)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0], [0.6, 0.8])
    np.testing.assert_allclose(out[1], [0.0, 0.0])  # zero vector stays zero, no NaN


def test_full_size_is_normalized():
    out = truncate_and_normalize(np.array([[3.0, 4.0]]), truncate_dim=None)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0])


def test_auto_dtype():
    assert resolve_dtype("auto", "cuda", "x", bf16_supported=True) == torch.bfloat16
    assert resolve_dtype("auto", "cuda", "x", bf16_supported=False) == torch.float32
    assert resolve_dtype("auto", "cpu", "x", bf16_supported=False) == torch.float32
    assert resolve_dtype(None, "cpu", "x", bf16_supported=False) is None


def test_encode_halves_batch_on_gpu_out_of_memory():
    class FakeModel:
        def __init__(self):
            self.batch_sizes = []

        def encode(self, sentences, batch_size, **kwargs):
            self.batch_sizes.append(batch_size)
            if batch_size > 16:
                raise torch.cuda.OutOfMemoryError
            return np.ones((len(sentences), 4))

    model = FakeModel()
    out = encode(model, ["a", "b"], OmegaConf.create({"truncate_dim": None}), batch_size=64)
    assert model.batch_sizes == [64, 32, 16]
    assert out.shape == (2, 4)
    assert out.dtype == np.float32


def test_encode_reraises_at_batch_size_one():
    class AlwaysOutOfMemory:
        def encode(self, sentences, batch_size, **kwargs):
            raise torch.cuda.OutOfMemoryError

    with pytest.raises(torch.cuda.OutOfMemoryError):
        encode(AlwaysOutOfMemory(), ["a"], OmegaConf.create({"truncate_dim": None}), batch_size=2)


def test_gemma_refuses_float16():
    with pytest.raises(ValueError, match="float16"):
        resolve_dtype("float16", "cuda", "google/embeddinggemma-2", bf16_supported=True)


def test_model_path():
    assert resolve_model_path("google/embeddinggemma-2", None) == "google/embeddinggemma-2"
    path = resolve_model_path("google/embeddinggemma-2", "/models")
    assert path.replace("\\", "/") == "/models/google/embeddinggemma-2"
