import time
from pathlib import Path

import polars as pl
from bertopic import BERTopic
from dvclive import Live
from omegaconf import OmegaConf
from polars import col as c

from job_post_topic_modelling.utils.miscellaneous import try_inter

try_inter()

from job_post_topic_modelling.utils.data_io import (  # noqa: E402
    load_pretrained_embeddings,
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
    texts = pl.read_parquet(data_dir / "texts.parquet")
    documents = texts["text"].to_list()
    topic_model = BERTopic.load(models_dir / "bertopic_model")
    embeddings = load_pretrained_embeddings(data_dir / "embeddings", nobs=None)

    # Predict topics on all documents
    print("Predicting topics on all docs...")
    topics, probs = topic_model.transform(
        documents,embeddings = embeddings
    )

    # topic_dict = topic_model.get_topics()
    texts = texts.with_row_index("row_nr").with_columns(
        pl.Series("predicted_topic", topics),
        pl.Series("topic_probability", probs),
        ann_id=c.label.str.extract(r"^(\d+)_s", 1),
        training_data = (pl.col("row_nr") < full_par.train.settings.nobs)
    ).drop("row_nr")

    print("Saving results...")
    # Save results
    texts.write_parquet(output_dir / "predicted_topics.parquet")

    print("aggregate to ann_id/job add level")
    topics_agg = texts.group_by("ann_id", "predicted_topic").agg(
        (pl.lit(1) - (pl.lit(1) - c("topic_probability")).product()).alias("topic_probability"),
    )
    print("make into wide format")
    topics_wide = (
        (
            topics_agg.sort("predicted_topic").pivot(
                values="topic_probability",
                index="ann_id",
                on="predicted_topic",
                aggregate_function=None,
            )
        )
        .fill_null(0.0)
        .select("ann_id", pl.all().exclude("ann_id").name.prefix("topic_"))
    )

    print("Saving aggregated results...")
    topics_wide.write_parquet(output_dir / "predicted_topics_agg.parquet")

    # Wrap up
    stop = time.time()
    hours = (stop - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

    # Log metrics using DVCLive
    with Live(dir=str(output_dir), cache_images=True, resume=True) as live:
        live.log_metric(f"{Path(__file__).name}", f"{hours:.2f} hours", plot=False)
