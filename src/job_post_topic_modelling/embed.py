import os
import time
from pathlib import Path

import numpy as np
from dvclive import Live
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from job_post_topic_modelling.utils.miscellaneous import print_params
from job_post_topic_modelling.utils.data_io import load_data


def get_embedding_model_name(embedding_model_name: str):
    """
    Get the embedding model name, checking if running on STATA server.
    This is because the star server needs a local path
    """

    # Check if running on STATA server, if yes set up path to load the correct SentenceTransformer
    if os.environ.get("CONDA_DEFAULT_ENV") in ["job_post_topic_modelling"]:
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


if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"
    embeddings_path = data_dir / "embeddings.npy"
    reduced_embeddings_path = data_dir / "reduced_embeddings.npy"

    # Load parameters
    par = OmegaConf.load(params_path).embed

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()
    print_params(par)

    # Load
    print("Loading data...")
    documents = load_data(
        str(data_dir / "texts.parquet"), text_col=par.text_col if hasattr(par, "text_col") else "text"
    )

    # Compute embeddings
    print("Embedding documents...")

    if par.model.use_model2vec:
        # Not tested for star server yet
        embedding_model_name = get_embedding_model_name(par.model.embedding_model)
        static_embedding = StaticEmbedding.from_distillation(
            embedding_model_name, device=par.settings.device, pca_dims=par.model.pca_dims
        )
        sentence_model = SentenceTransformer(modules=[static_embedding])
    else:
        sentence_model = get_embedding_model(par.model.embedding_model)

    embeddings = sentence_model.encode(
        documents,
        show_progress_bar=par.settings.show_progress_bar,
        batch_size=par.settings.batch_size,
        device=par.settings.device,
        num_workers=par.settings.num_workers,
    )

    # Reduce embedding dimensions
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine", unique=True).fit_transform(
    #    embeddings
    # )

    # Save
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")
    # np.save(reduced_embeddings_path, reduced_embeddings)
    # print(f"Saved reduced_embeddings to {reduced_embeddings_path}")

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True, resume=True) as live:
        # Log metrics
        live.log_metric(f"{Path(__file__).name}", f"{hours:.2f} hours", plot=False)
