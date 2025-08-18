from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer

from job_post_topic_modelling.utils.data_io import load_data

if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"
    embeddings_path = data_dir / "embeddings.npy"

    # Load parameters
    par = OmegaConf.load(params_path).predict

    # Load texts using the same function as predict.py
    documents = load_data(
        str(data_dir / "texts.parquet"), text_col=par.text_col if hasattr(par, "text_col") else "text"
    )

    # Compute embeddings using the model from parameters
    print("Loading sentence transformer model...")
    sentence_model = SentenceTransformer(par.embedding.embedding_model)
    print("Encoding documents...")
    embeddings = sentence_model.encode(documents, show_progress_bar=True)

    # Save embeddings
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")
