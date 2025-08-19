from pathlib import Path

import matplotlib.pyplot as plt
from bertopic import BERTopic
from dvclive import Live
from matplotlib.figure import Figure
from omegaconf import OmegaConf

from job_post_topic_modelling.utils.interactive import try_inter

try_inter()
from job_post_topic_modelling.utils.data_io import load_data  # noqa: E402
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402
from job_post_topic_modelling.utils.log_html import log_html  # noqa: E402


class InvalidInputFileError(Exception):
    def __init__(self) -> None:
        super().__init__("Input file must contain a list of lists.")


def load_model(model_path: str | Path) -> object:
    """
    Load a model from a file.

    Args:
        model_path (str | Path): Path to the model file.

    Returns:
        object: The loaded model.
    """
    return BERTopic.load(model_path)


def create_top_words_fig(model) -> Figure:
    """
    Create a picture of  string with the top words for each topic.

    Args:
        top_words (dict): Dictionary with topic numbers as keys and lists of top words as values.
        png_path (Path): Path to save the generated image.

    Returns:
        None
    """
    topic_info = model.get_topic_info()
    top_words = topic_info.set_index("Topic")["Representation"].to_dict()

    text = "# Top Words per Topic\n"
    for topic_n, words_list in top_words.items():
        words_str = ", ".join(words_list)
        text += f"# Topic {topic_n + 1}: {words_str}\n"

    fig = plt.figure(figsize=(8, 10))
    plt.text(0.01, 0.99, text, fontsize=12, family="monospace", va="top", ha="left", wrap=True)
    plt.axis("off")

    return fig


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"

    # Load parameters
    par = OmegaConf.load(params_path).evaluate

    # load
    documents = load_data(data_dir / "texts.parquet", text_col="text")
    topic_model = load_model(models_dir / "bertopic_model")

    with Live(dir=str(output_dir), cache_images=True, resume=True) as live:
        topic_info = topic_model.get_topic_info()
        output_file = output_dir / "topic_info.csv"
        topic_info.to_csv(output_file)
        live.log_artifact(output_file, type="dataset")

        live.log_metric("#Topics", len(topic_info), plot=False)

        fig = create_top_words_fig(topic_model)
        live.log_image("top_words.png", fig)

        topics_fig = topic_model.visualize_topics()
        log_html(live, "topics_fig.png", topics_fig)

        heatmap_fig = topic_model.visualize_heatmap()
        log_html(live, "heatmap_fig.png", heatmap_fig)

        hierarchy_fig = topic_model.visualize_hierarchy()
        log_html(live, "hierarchy_fig.png", hierarchy_fig)

        # family
        similar_topics, similarity = topic_model.find_topics(par.settings.similarity_phrase, top_n=par.settings.top_n)
        barchart_fig = topic_model.visualize_barchart(
            similar_topics,
            top_n_topics=par.settings.top_n,
            title=f"Similar Topics to '{par.settings.similarity_phrase}'",
        )
        log_html(live, "barchart_fig.png", barchart_fig)
