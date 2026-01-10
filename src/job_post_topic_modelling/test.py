import re
from typing import Dict, List, Optional
from ctransformers import AutoModelForCausalLM
from pathlib import Path
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from dvclive import Live
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline

# from celer import LassoCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from job_post_topic_modelling.utils.miscellaneous import print_params, try_inter

try_inter()
from job_post_topic_modelling.embed import get_embedding_model, get_embedding_model_name  # noqa: E402
from job_post_topic_modelling.train import get_embedding_model_cpu, load_pretrained_embeddings  # noqa: E402
from job_post_topic_modelling.utils.data_io import (  # noqa: E402
    load_danish_stop_words,
)
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402
from job_post_topic_modelling.utils.log_html import log_html  # noqa: E402


os.environ["CUDA_VISIBLE_DEVICES"] = ""

class InvalidInputFileError(Exception):
    def __init__(self) -> None:
        super().__init__("Input file must contain a list of lists.")


class UnknownModelError(Exception):
    """Exception raised when an unknown model is encountered."""

    def __init__(self, model_name: str):
        super().__init__(f"Unknown model: {model_name}")


def load_model(model_path: str | Path) -> object:
    """
    Load a model from a file.

    Args:
        model_path (str | Path): Path to the model file.

    Returns:
        object: The loaded model.
    """
    return BERTopic.load(model_path)


def add_representative_docs(
    topic_model: BERTopic, documents: list[str], nr_samples: int = 2000, nr_repr_docs: int = 5
) -> None:
    """
    Add representative documents to a BERTopic topic_model.

    Args:
        topic_model (BERTopic): The BERTopic topic_model.
        documents (list[str]): List of documents used for topic topic_modeling.

    Returns:
        None
    """
    docs_df = pl.DataFrame({
        "Document": documents[: len(topic_model.topics_)],
        "Topic": topic_model.topics_,
        "ID": range(len(topic_model.topics_)),
    }).to_pandas()

    topic_model.representative_docs_, _, _, _ = topic_model._extract_representative_docs(
        topic_model.c_tf_idf_,
        docs_df,
        topic_model.topic_representations_,
        nr_samples=nr_samples,
        nr_repr_docs=nr_repr_docs,
    )


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


def get_cTFIDF_model(par: OmegaConf):
    args = {k: v for k, v in par.c_TF_IDF.items() if k != "model"}
    if par.c_TF_IDF.model == "c_TF_IDF":
        ctfidf_model = ClassTfidfTransformer(**args)
    else:
        raise UnknownModelError(par.c_TF_IDF.model)
    return ctfidf_model


def get_representation_model(par: OmegaConf):
    args = {k: v for k, v in par.representation.items() if k != "model"}
    if par.representation.model == "KeyBERTInspired":
        representation_model = KeyBERTInspired(**args)
    elif par.representation.model == "MMR":
        representation_model = MaximalMarginalRelevance(**args)
    else:
        raise UnknownModelError(par.representation.model)
    return representation_model


def get_vectorizer(par: OmegaConf, stop_words=None):
    args = {k: v for k, v in par.vectorizer.items() if k != "model"}
    if "ngram_range" in args:
        args["ngram_range"] = tuple(args["ngram_range"])
    if par.vectorizer.model == "CountVectorizer":
        vectorizer_model = CountVectorizer(stop_words=stop_words, **args)
    else:
        raise UnknownModelError(par.vectorizer.model)
    return vectorizer_model


def load_click_shares() -> pl.DataFrame:
    # get username
    username = os.popen("whoami").read().strip()  # noqa: S607 S605

    folder_path = Path(f"/home/{username}@PROD.SITAD.DK/code/jobads/src/dgp/prep_clicks_for_dvc/output")

    click_shares = pl.read_parquet(folder_path / "ads_clicks_agg.parquet")
    return click_shares


def demean(df, var_list, by_var):
    """
    Demean variables in var_list by by_var.
    Simply replaces the variables in var_list with their demeaned versions.

    """
    group_means = df.group_by(by_var).agg([pl.col(v).mean().alias(f"{v}_mean") for v in var_list])
    df = df.drop(cs.ends_with("_mean")).join(group_means, on=by_var, how="left")

    df = df.with_columns([(pl.col(v) - pl.col(f"{v}_mean")).alias(v) for v in var_list])

    return df


def linear_lasso_cv_oos(
    df,
    outcome,
    predictors,
    test_size=0.2,
    n_folds=5,
    binary_cut=None,
    use_sample_weight=False,
    weight_var="click_count",
    standardize=True,
    is_sparse=True,
    random_state=123521,
):
    """
    df: Pandas or Polars DataFrame
    outcome: continuous outcome column
    predictors: list of predictor names
    Performs K-fold CV on the training data to choose alpha.
    """

    # Extract data
    if hasattr(df, "select"):  # Polars
        na_filter = pl.col(outcome).is_not_null()
        X = df.filter(na_filter).select(predictors).to_numpy()
        y = df.filter(na_filter)[outcome].to_numpy()
        if use_sample_weight:
            w = df.filter(na_filter)[weight_var].to_numpy()
    else:  # Pandas
        na_filter = df[outcome].notna()
        X = df.loc[na_filter, predictors].to_numpy()
        y = df.loc[na_filter, outcome].to_numpy()

        if use_sample_weight:
            w = df.loc[na_filter, weight_var].to_numpy()

    if binary_cut is not None:
        X = (binary_cut <= X).astype(int)

    if is_sparse:
        X = csr_matrix(X)

    if standardize:
        scaler = StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)

    # Train/test split
    if use_sample_weight:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, w, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        w_train = None
        w_test = None

    # Cross-validated Lasso
    model = LassoCV(
        cv=KFold(n_splits=n_folds, shuffle=True, random_state=random_state),
        max_iter=5000,
        n_jobs=4,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    # Extract coefficients
    coef = model.coef_
    selected = [v for v, coefi in zip(predictors, coef) if coefi != 0]

    # Predict OOS
    y_hat = model.predict(X_test)

    # OOS metrics
    if use_sample_weight:
        mse = mean_squared_error(y_test, y_hat, sample_weight=w_test)
        mae = mean_absolute_error(y_test, y_hat, sample_weight=w_test)
        # sklearn's r2_score *does* support sample_weight
        r2 = r2_score(y_test, y_hat, sample_weight=w_test)
    else:
        mse = mean_squared_error(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        r2 = r2_score(y_test, y_hat)

    return {
        "model": model,
        "chosen_alpha": model.alpha_,
        "n_selected": len(selected),
        "n_possible": len(predictors),
        "coefficients": dict(zip(predictors, coef)),
        "selected_predictors": selected,
        "y_test": y_test,
        "y_hat": y_hat,
        "oos_mse": mse,
        "oos_r2": r2,
        "oos_mae": mae,
        "n_obs": len(y),
    }


def R2_dicts_to_fig(
    dict_list,
    row_names=None,
    columns=None,
    float_format="%.4f",
):
    """
    Convert list of dicts into matplotlib fig that can be logged

    Parameters
    ----------
    dict_list : list of dict
        One dictionary per row.
    row_names : list of str, optional
        Names for the rows.
    columns : dict, optional
        Mapping: original_key -> label_to_show_in_table
        Example: {"tjur_r2": "Tjur $R^2$", "n_selected": "Selected"}
        Only these keys will be included in the table, in this order.
    float_format : str
        Format for floating-point values, e.g. "%.4f"
    """
    if row_names is None:
        row_names = [f"Model {i + 1}" for i in range(len(dict_list))]
    if columns is None:
        raise ValueError("columns must be provided")  # noqa: TRY003

    output_text = ""
    for row_name, result_dict in zip(row_names, dict_list):
        output_text += f"# {row_name}\n"
        for key, label in columns.items():
            value = result_dict.get(key, "N/A")
            value_str = float_format % value if isinstance(value, float) else str(value)
            output_text += f"#   {label}: {value_str}\n"
        output_text += "\n"

    fig = plt.figure(figsize=(10, 12))
    plt.text(0.01, 0.99, output_text, fontsize=14, family="monospace", va="top", ha="left", wrap=True)
    plt.axis("off")

    return fig


if __name__ == "__main__":
    # Define file paths
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    models_dir = project_root / "models"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"
    seed = 1230532
    # Load parameters
    full_par = OmegaConf.load(params_path)
    par = full_par.evaluate
    par_train = full_par.train
    embedding_model_name = full_par.embed.model.embedding_model

    # Process
    print(f"Starting {Path(__file__).name}")
    start = time.time()
    print_params(full_par)

    # load
    print("Loading data...")
    texts = pl.read_parquet(data_dir / "texts.parquet")
    if par_train.settings.nobs is not None:
        texts = texts.head(par_train.settings.nobs)
    documents = texts["text"].to_list()

    embeddings = load_pretrained_embeddings(data_dir / "embeddings", nobs=par_train.settings.nobs)

    print("Loading model")
    stop_words = load_danish_stop_words(data_dir / "stopwords-da.json")
    embedding_model_name = full_par.embed.model.embedding_model
    if par.settings.use_cpu:
        embedding_model = get_embedding_model_cpu(embedding_model_name)
    else:
        embedding_model = get_embedding_model(embedding_model_name)

    ctfidf_model = get_cTFIDF_model(par)
    representation_model = get_representation_model(par)

    vectorizer_model = get_vectorizer(par, stop_words=stop_words)

    topic_model = BERTopic.load(models_dir / "bertopic_model", embedding_model=embedding_model)

def _clean_label(text: str) -> str:
    text = (text or "").strip()
    # Keep first line only
    text = text.splitlines()[0].strip()
    # Strip surrounding quotes
    text = re.sub(r'^\s*["“”\']+|["“”\']+\s*$', "", text).strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text

def label_topics_with_ctransformers(
    topic_model,
    llm,  # your ctransformers model (callable: llm(prompt, **gen_kwargs))
    prompt_template: str,
    top_n_words: int = 12,
    use_docs: bool = True,
    nr_docs: int = 3,
    gen_kwargs: Optional[dict] = None,
) -> Dict[int, str]:
    """
    Returns: {topic_id: "Nice Danish label"}
    """
    gen_kwargs = gen_kwargs or dict(
        max_new_tokens=16,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    labels: Dict[int, str] = {}
    print('Generating topic labels...')
    # Iterate topics (skip outlier topic -1)
    for topic_id in sorted(topic_model.get_topics().keys()):
        if topic_id == -1:
            continue

        # Keywords
        kw = [w for w, _ in topic_model.get_topic(topic_id)[:top_n_words]]
        keywords = ", ".join(kw)

        # Representative docs (optional; API differs a bit across versions)
        docs_block = ""
        if use_docs:
            docs: List[str] = []
            try:
                docs = topic_model.get_representative_docs(topic=topic_id)  # common signature
            except TypeError:
                try:
                    docs = topic_model.get_representative_docs(topic_id)  # older signature
                except Exception:
                    docs = []
            except Exception:
                docs = []

            docs = (docs or [])[:nr_docs]
            if docs:
                docs_block = "\n".join([f"- {d[:400].replace('\\n',' ')}" for d in docs])

        prompt = (
            prompt_template
            .replace("[KEYWORDS]", keywords)
            .replace("[DOCUMENTS]", docs_block)
        )
        # Generate
        out = llm(prompt, **gen_kwargs)
        label = _clean_label(out)
        print(f"Topic {topic_id}: {label}")

        # Fallback if model returns nothing useful
        if not label:
            label = " ".join(kw[:3])

        labels[topic_id] = label

    return labels
prompt_template = """[INST]
Du hjælper med at give korte emne-etiketter på dansk.

Nøgleord: [KEYWORDS]

Eksempel-dokumenter:
[DOCUMENTS]

Returnér KUN en kort etiket (2-5 ord). Ingen ekstra tekst, ingen citationstegn.
[/INST]
"""

model_path = Path(r'/home/b281467@PROD.SITAD.DK/code/help/installations/Mistral-7B-Instruct-v0.3-GGUF')
model_filename = 'Mistral-7B-Instruct-v0.3-Q6_K.gguf'
llm = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        model_file=model_filename,
        model_type="mistral",
        gpu_layers=0,
        hf=False
    )

labels = label_topics_with_ctransformers(
    topic_model=topic_model,
    llm=llm,  # your ctransformers model object
    prompt_template=prompt_template,
    top_n_words=12,
    use_docs=True,
    nr_docs=3,
)

labels