"""Load sentence embedding models from config and encode sentences with them."""

import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding

PROMPT_NAME = "clustering"
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def resolve_device(device: str | None) -> str:
    """Return the requested device, or cuda when available (else cpu) if none was requested."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(dtype: str | None, device: str, model_name: str, bf16_supported: bool) -> torch.dtype | None:
    """Turn the config dtype into a torch dtype; None means "library default".

    "auto" is bfloat16 on a GPU that supports it and float32 otherwise.
    EmbeddingGemma is never run in float16.
    """
    if dtype is None:
        return None
    if dtype == "auto":
        return torch.bfloat16 if device == "cuda" and bf16_supported else torch.float32
    if dtype == "float16" and "embeddinggemma" in model_name.lower():
        raise ValueError(f"{model_name} must not run in float16, use bfloat16 or float32")  # noqa: TRY003
    return DTYPES[dtype]


def resolve_model_path(model_name: str, model_dir: str | None) -> str:
    """Return the hub name, or the local folder holding the copied model when model_dir is set."""
    if model_dir is None:
        return model_name
    path = Path(model_dir) / model_name
    if not path.exists():
        message = f"embed.settings.model_dir is set but the model folder does not exist: {path}"
        raise FileNotFoundError(message)
    return str(path)


def load_embedding_model(model_par: DictConfig, settings: DictConfig, device: str | None = None) -> SentenceTransformer:
    """Build the SentenceTransformer described by the model config.

    `device` overrides settings.device (train and predict pass "cpu").
    With settings.model_dir set (STAR), the model is loaded offline from that folder.
    """
    model_dir = settings.get("model_dir")
    if model_dir:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = resolve_device(device or settings.device)
    path = resolve_model_path(model_par.embedding_model, model_dir)

    if model_par.use_model2vec:
        static_embedding = StaticEmbedding.from_distillation(path, device=device, pca_dims=model_par.pca_dims)
        return SentenceTransformer(modules=[static_embedding])

    bf16_supported = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = resolve_dtype(model_par.dtype, device, model_par.embedding_model, bf16_supported)
    prompt = model_par.prompt
    model = SentenceTransformer(
        path,
        device=device,
        truncate_dim=model_par.truncate_dim,
        model_kwargs={"dtype": dtype} if dtype else None,
        config_kwargs=OmegaConf.to_container(model_par.config_kwargs) if model_par.config_kwargs else None,
        # The default prompt applies to every encode call, including BERTopic's own calls on seed words.
        prompts={PROMPT_NAME: prompt} if prompt else None,
        default_prompt_name=PROMPT_NAME if prompt else None,
    )
    if model_par.max_seq_length:
        model.max_seq_length = model_par.max_seq_length
    return model


def truncate_and_normalize(embeddings: np.ndarray, truncate_dim: int | None) -> np.ndarray:
    """Keep the first truncate_dim dimensions (Matryoshka) and scale each row to unit length.

    All-zero rows stay zero instead of becoming NaN.
    """
    if truncate_dim:
        embeddings = embeddings[:, :truncate_dim]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def encode(
    model: SentenceTransformer, sentences: list[str], model_par: DictConfig, batch_size: int, **encode_kwargs
) -> np.ndarray:
    """Embed sentences into float32, truncated and normalized vectors.

    On GPU out-of-memory the batch size is halved and the call retried; at batch size 1 the error is raised.
    """
    while True:
        try:
            embeddings = model.encode(sentences, batch_size=batch_size, convert_to_numpy=True, **encode_kwargs)
            break
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size //= 2
            print(f"GPU out of memory, retrying with batch_size={batch_size}")
    return truncate_and_normalize(embeddings, model_par.truncate_dim).astype(np.float32)
