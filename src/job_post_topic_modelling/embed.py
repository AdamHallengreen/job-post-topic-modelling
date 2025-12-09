import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

from job_post_topic_modelling.utils.data_io import load_data
from job_post_topic_modelling.utils.miscellaneous import print_params


def get_embedding_model_name(embedding_model_name: str):
    """
    Get the embedding model name, checking if running on STATA server.
    This is because the star server needs a local path
    """

    # Check if running on STATA server, if yes set up path to load the correct SentenceTransformer
    if os.environ.get("CONDA_DEFAULT_ENV") in ["job_post_topic_modelling", "job_rapids313"]:
        user = os.popen("whoami").read().strip()  # noqa: S605, S607
        # Optional: force strict offline behavior
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if embedding_model_name in [
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]:
            pref = rf"/home/{user}@PROD.SITAD.DK/code/help/installations/"
        else:
            raise ValueError(f"Model {embedding_model_name} not recognized in STATA server setup.")  # noqa: TRY003
    else:
        pref = ""
    return pref + embedding_model_name


def get_embedding_model(embedding_model_name: str):
    """
    Get the embedding model name, checking if running on STATA server.
    This is because the star server needs a local path
    """

    return SentenceTransformer(get_embedding_model_name(embedding_model_name))


def embed_in_shards(documents, sentence_model, embeddings_path, shard_size=1_000_000, **encode_kwargs):
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
        shard_embeddings = sentence_model.encode(
            documents[start_idx:end_idx],
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
    documents = load_data(
        str(data_dir / "texts.parquet"), text_col=par.text_col if hasattr(par, "text_col") else "text"
    )

    # Compute embeddings
    print("Embedding and saving documents...")

    if par.model.use_model2vec:
        # Not tested for star server yet
        embedding_model_name = get_embedding_model_name(par.model.embedding_model)
        static_embedding = StaticEmbedding.from_distillation(
            embedding_model_name, device=par.settings.device, pca_dims=par.model.pca_dims
        )
        sentence_model = SentenceTransformer(modules=[static_embedding])
    else:
        sentence_model = get_embedding_model(par.model.embedding_model)

    embed_in_shards(
        documents,
        sentence_model,
        embeddings_path,
        shard_size=par.settings.shard_size,
        show_progress_bar=par.settings.show_progress_bar,
        batch_size=par.settings.batch_size,
        device=par.settings.device,
        num_workers=par.settings.num_workers,
    )

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    with ((output_dir / "metrics") / "embed.json").open("w") as f:
        json.dump({f"{Path(__file__).name}": f"{hours:.2f} hours"}, f, indent=4)
