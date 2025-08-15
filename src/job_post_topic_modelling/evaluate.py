from pathlib import Path

from bertopic import BERTopic

from job_post_topic_modelling.utils.interactive import try_inter

try_inter()

from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402


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


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    models_dir = project_root / "models"
    output_dir = project_root / "output"
    metrics = project_root / "metrics.yaml"
    top_words_picture = output_dir / "top_words.png"

    # Process
    model = load_model(models_dir / "bertopic_model")

    # output
    topic_info = model.get_topic_info()
    output_file = output_dir / "topic_info.csv"
    topic_info.to_csv(output_file)
