import time
from pathlib import Path

import numpy as np
from dvclive import Live
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

from job_post_topic_modelling.utils.data_io import load_data

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"
    embeddings_path = data_dir / "embeddings.npy"

    # Load parameters
    par = OmegaConf.load(params_path).embed

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()

    # Load
    documents = load_data(
        str(data_dir / "texts.parquet"), text_col=par.text_col if hasattr(par, "text_col") else "text"
    )

    # Compute embeddings
    print("Encoding documents...")
    if par.model.use_model2vec:
        static_embedding = StaticEmbedding.from_distillation(
            par.model.embedding_model, device=par.settings.device, pca_dims=par.model.pca_dims
        )
        sentence_model = SentenceTransformer(modules=[static_embedding])
    else:
        sentence_model = SentenceTransformer(par.model.embedding_model)
    embeddings = sentence_model.encode(
        documents,
        show_progress_bar=par.settings.show_progress_bar,
        batch_size=par.settings.batch_size,
        device=par.settings.device,
        num_workers=par.settings.num_workers,
    )

    # Save
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True, resume=True) as live:
        # Log metrics
        live.log_metric(f"{Path(__file__).name}", f"{hours:.2f} hours", plot=False)
