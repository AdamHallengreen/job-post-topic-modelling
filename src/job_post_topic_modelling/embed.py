import json
import shutil
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from job_post_topic_modelling.embedding_models import encode, load_embedding_model
from job_post_topic_modelling.utils.data_io import load_data
from job_post_topic_modelling.utils.miscellaneous import print_params


def embedding_cost(
    n_sentences: int, seconds: float, vector_dim: int, n_projected: int = 56_000_000
) -> dict[str, float]:
    """Speed and storage cost of an embedding model, assuming float32 vectors (4 bytes per number)."""
    bytes_per_vector = vector_dim * 4
    return {
        "sentences_per_second": n_sentences / seconds,
        "vector_dim": vector_dim,
        "bytes_per_vector": bytes_per_vector,
        "projected_storage_gb": round(n_projected * bytes_per_vector / 1e9, 3),
    }


def embed_in_shards(
    documents, sentence_model, model_par, embeddings_path, batch_size, shard_size=1_000_000, **encode_kwargs
):
    """
    Embed documents in shards to avoid memory issues.
    """
    # clear folder
    if embeddings_path.exists():
        shutil.rmtree(embeddings_path)
    embeddings_path.mkdir(parents=True, exist_ok=True)

    num_docs = len(documents)
    i = 1
    for start_idx in range(0, num_docs, shard_size):
        print(f"Processing shard {i}")
        end_idx = min(start_idx + shard_size, num_docs)
        print(f"Embedding documents {start_idx} to {end_idx} out of {num_docs}...")
        shard_embeddings = encode(
            sentence_model,
            documents[start_idx:end_idx],
            model_par,
            batch_size=batch_size,
            **encode_kwargs,
        )
        np.save(embeddings_path / f"embeddings_shard_{i}.npy", shard_embeddings)
        i += 1


if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"
    embeddings_path = data_dir / "embeddings"
    reduced_embeddings_path = data_dir / "reduced_embeddings.npy"

    # Load parameters
    full_par = OmegaConf.load(params_path)
    par = full_par.embed

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()
    print_params(full_par)

    # Load
    print("Loading data...")
    sentences = load_data(data_dir / "sentences.parquet", text_col="text")

    # Compute embeddings (row i belongs to sentence_id i)
    print("Embedding and saving unique sentences...")
    sentence_model = load_embedding_model(par.model, par.settings)

    embed_start = time.time()
    embed_in_shards(
        sentences,
        sentence_model,
        par.model,
        embeddings_path,
        batch_size=par.settings.batch_size,
        shard_size=par.settings.shard_size,
        show_progress_bar=par.settings.show_progress_bar,
        device=par.settings.device,
    )
    embed_seconds = time.time() - embed_start

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    with ((output_dir / "metrics") / "embed.json").open("w") as f:
        metrics = {f"{Path(__file__).name}": f"{hours:.2f} hours"}
        metrics.update(
            embedding_cost(len(sentences), embed_seconds, vector_dim=sentence_model.get_embedding_dimension())
        )
        json.dump(metrics, f, indent=4)
