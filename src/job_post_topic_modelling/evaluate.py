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

# from sklearn.linear_model import LassoCV
from celer import LassoCV
from dvclive import Live
from matplotlib.figure import Figure
from omegaconf import OmegaConf
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from job_post_topic_modelling.utils.miscellaneous import print_params, try_inter

try_inter()
from job_post_topic_modelling.embed import get_embedding_model, get_embedding_model_name  # noqa: E402
from job_post_topic_modelling.train import load_pretrained_embeddings  # noqa: E402
from job_post_topic_modelling.utils.data_io import (  # noqa: E402
    load_danish_stop_words,
)
from job_post_topic_modelling.utils.find_project_root import find_project_root  # noqa: E402
from job_post_topic_modelling.utils.log_html import log_html  # noqa: E402


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
    if par.representation.model == "MMR":
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
        n_jobs=10,
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

    fig = plt.figure(figsize=(8, 10))
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
    embedding_model = get_embedding_model(embedding_model_name)

    ctfidf_model = get_cTFIDF_model(par)
    representation_model = get_representation_model(par)
    vectorizer_model = get_vectorizer(par, stop_words=stop_words)

    topic_model = BERTopic.load(models_dir / "bertopic_model", embedding_model=embedding_model)

    print("Updating topic representation...")
    topic_model.update_topics(
        documents[: par_train.settings.nobs],
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
    )

    # Save model
    print("Saving model with updated topic representation...")
    topic_model.save(
        output_dir / "bertopic_model_representation",
        serialization="safetensors",
        save_ctfidf=False,  # True, # There is some error here for TRUE
        save_embedding_model=get_embedding_model_name(embedding_model_name),
    )

    print("Logging metrics from previous stages...")
    stages = ["prepare", "embed", "train", "predict"]
    with Live(dir=str(output_dir), cache_images=True, resume=False) as live:
        for stage in stages:
            print(stage)
            stage_metrics_path = (output_dir / "metrics") / f"{stage}.json"
            with stage_metrics_path.open("r") as f:
                stage_metrics = json.load(f)
            print(stage_metrics)
            for key, value in stage_metrics.items():
                live.log_metric(key, value, plot=False)

    # If running on star data calculate out-of-sample prediction of click-shares
    if full_par.prepare.star.usestar == 1:
        print("Calculating out-of-sample prediction of click-shares...")
        click_shares = load_click_shares()
        topics = pl.read_parquet(output_dir / "predicted_topics_agg.parquet")

        log_shares = [
            pl.when(pl.col(f"apply_share{suf}") == 0)
            .then(None)  # set zeros to null
            .otherwise(pl.col(f"apply_share{suf}"))
            .log()
            .alias(f"l_apply_share{suf}")
            for suf in ["", "_male", "_fem"]
        ]
        df_merged = topics.join(click_shares, on="ann_id", how="inner").with_columns(*log_shares)

        predictors = topics.select(cs.starts_with("p_topic_")).columns
        d_predictors = topics.select(cs.starts_with("d_topic_")).columns

        is_sparse = not full_par.predict.settings.full_p_dist

        results_share_cv_log = linear_lasso_cv_oos(
            df=df_merged,
            outcome="l_apply_share",
            predictors=predictors,
            random_state=seed,
            is_sparse=is_sparse,
        )

        # residualized by occupation group
        df_merged_resid_gr = demean(
            df_merged, var_list=[f"l_apply_share{suf}" for suf in ["", "_male", "_fem"]] + predictors, by_var="faggrid"
        )
        results_share_cv_log_resid_gr = linear_lasso_cv_oos(
            df=df_merged_resid_gr,
            outcome="l_apply_share",
            predictors=predictors,
            random_state=seed,
            is_sparse=False,
        )

        # residualized by occupation area
        df_merged_resid_omr = demean(
            df_merged, var_list=[f"l_apply_share{suf}" for suf in ["", "_male", "_fem"]] + predictors, by_var="fagomrid"
        )
        results_share_cv_log_resid_omr = linear_lasso_cv_oos(
            df=df_merged_resid_omr,
            outcome="l_apply_share",
            predictors=predictors,
            random_state=seed,
            is_sparse=False,
        )

        if full_par.predict.settings.full_p_dist:
            d_results = []
            d_name = []
        else:
            results_share_cv_log_d = linear_lasso_cv_oos(
                df=df_merged,
                outcome="l_apply_share",
                predictors=d_predictors,
                random_state=seed,
                is_sparse=True,
            )
            d_results = [results_share_cv_log_d]
            d_name = ["Dummies"]

        results_share_cv_log_train = linear_lasso_cv_oos(
            df=df_merged.filter(pl.col("training_data") == 1),
            outcome="l_apply_share",
            predictors=predictors,
            random_state=seed,
            is_sparse=is_sparse,
        )
        results_share_cv_log_male = linear_lasso_cv_oos(
            df=df_merged,
            outcome="l_apply_share_male",
            predictors=predictors,
            random_state=seed,
            is_sparse=is_sparse,
        )

        results_share_cv_log_fem = linear_lasso_cv_oos(
            df=df_merged,
            outcome="l_apply_share_fem",
            predictors=predictors,
            random_state=seed,
            is_sparse=is_sparse,
        )

        print("Calculating for training topics")
        topics_train = pl.read_parquet(output_dir / "predicted_topics_agg_train.parquet")

        df_merged_train = topics_train.join(click_shares, on="ann_id", how="inner").with_columns(*log_shares)

        predictors = topics_train.select(cs.starts_with("p_topic_")).columns

        results_share_cv_log_train_topics = linear_lasso_cv_oos(
            df=df_merged_train,
            outcome="l_apply_share",
            predictors=predictors,
            random_state=seed,
            is_sparse=True,
        )

        print("Log Click Share OOS R2:")
        columns = {
            "n_possible": r"\# Possible",
            "n_selected": r"\# Selected",
            "oos_r2": "Out-of-sample R²",
            "n_obs": r"\# Obs",
        }

        output_fig = R2_dicts_to_fig(
            [
                results_share_cv_log,
                results_share_cv_log_resid_gr,
                results_share_cv_log_resid_omr,
                results_share_cv_log_train,
                results_share_cv_log_train_topics,
                results_share_cv_log_male,
                results_share_cv_log_fem,
                *d_results,
            ],
            row_names=[
                "All",
                "Residualized by group",
                "Residualized by area",
                "Topic training",
                "Topic training topics",
                "Men",
                "Women",
                *d_name,
            ],
            columns=columns,
            float_format="%.4f",
        )

        with Live(dir=str(output_dir), cache_images=True, resume=True) as live:
            live.log_metric("click_share_cv_log_r2", f"{results_share_cv_log['oos_r2']:.4f}", plot=False)
            live.log_metric("click_share_cv_log_n_selected", results_share_cv_log["n_selected"], plot=False)
            live.log_metric("click_share_male_cv_log_r2", f"{results_share_cv_log_male['oos_r2']:.4f}", plot=False)
            live.log_metric("click_share_male_cv_log_n_selected", results_share_cv_log_male["n_selected"], plot=False)
            live.log_metric("click_share_fem_cv_log_r2", f"{results_share_cv_log_fem['oos_r2']:.4f}", plot=False)
            live.log_metric("click_share_fem_cv_log_n_selected", results_share_cv_log_fem["n_selected"], plot=False)

            print("Saving table")
            live.log_image("click_share_R2.png", output_fig)

    print("Creating metrics and visualizations...")
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

        # documents_fig = topic_model.visualize_document_datamap(
        #     documents,
        #     reduced_embeddings=reduced_embeddings,
        #     datamap_kwds={"label_over_points": True, "dynamic_label_size": True},
        # )
        # log_html(live, "documents_fig.png", documents_fig)

        # family
        similar_topics, similarity = topic_model.find_topics(par.settings.similarity_phrase, top_n=par.settings.top_n)
        barchart_fig = topic_model.visualize_barchart(
            similar_topics,
            top_n_topics=par.settings.top_n,
            title=f"Similar Topics to '{par.settings.similarity_phrase}'",
        )
        log_html(live, "barchart_fig.png", barchart_fig)

        # Wrap up
        stop = time.time()
        hours = (stop - start) / 3600
        print(f"Finished {Path(__file__).name} in {hours:.2f} hours")

        live.log_metric(f"{Path(__file__).name}", f"{hours:.2f} hours", plot=False)
