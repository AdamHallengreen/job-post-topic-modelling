import time
from pathlib import Path

import polars as pl
from dvclive import Live
from omegaconf import OmegaConf

from job_post_topic_modelling.utils.miscellaneous import try_inter

try_inter()

from job_post_topic_modelling.evaluate import (  # noqa: E402
    get_cTFIDF_model,
    get_representation_model,
    get_vectorizer,
    load_model,
)
from job_post_topic_modelling.utils.data_io import (  # noqa: E402
    load_danish_stop_words,
)
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402
from job_post_topic_modelling.utils.miscellaneous import print_params  # noqa: E402

if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"

    # Load parameters
    full_par = OmegaConf.load(params_path)
    par = full_par.evaluate
    embedding_model_name = full_par.embed.model.embedding_model

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()
    print_params(full_par)

    # load
    print("Loading data...")
    texts = pl.read_parquet(data_dir / "texts.parquet").sample(full_par.train.settings.nobs + 1000)
    documents = texts["text"].to_list()
    topic_model = load_model(models_dir / "bertopic_model")
    stop_words = load_danish_stop_words(data_dir / "stopwords-da.json")
    # reduced_embeddings = load_pretrained_embeddings(data_dir / "reduced_embeddings.npy")

    # Choose models
    vectorizer_model = get_vectorizer(par, stop_words=stop_words)
    ctfidf_model = get_cTFIDF_model(par)
    representation_model = get_representation_model(par)

    # Adjust topic representation
    print("Updating topic representation...")
    topic_model.update_topics(
        documents[: full_par.train.settings.nobs],
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
    )

    # Predict topics on all documents
    print("Predicting topics on all docs...")
    topics, probs = topic_model.transform(
        documents,
    )

    # topic_dict = topic_model.get_topics()
    texts = texts.with_columns(
        pl.Series("predicted_topic", topics),
        pl.Series("topic_probability", probs),
    )

    print("Saving results...")
    # Save results
    texts.write_parquet(output_dir / "predicted_topics.parquet")

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True, resume=True) as live:
        live.log_metric(f"{Path(__file__).name}", f"{hours:.2f} hours", plot=False)
