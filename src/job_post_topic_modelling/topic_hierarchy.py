import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from bertopic import BERTopic
from omegaconf import OmegaConf

from job_post_topic_modelling.utils.find_project_root import find_project_root
from job_post_topic_modelling.utils.miscellaneous import print_params, try_inter

try_inter()


def build_hierarchy_levels(
    topic_model: BERTopic,
    documents: list[str],
    n_levels: int,
) -> pd.DataFrame:
    """
    Build a DataFrame mapping each leaf topic to its cluster name at each hierarchy level.

    Uses BERTopic's hierarchical_topics to get the merge tree, then cuts the
    dendrogram at n_levels evenly spaced distance thresholds.

    Returns a pandas DataFrame with columns: Topic, step_1, step_2, ..., step_N
    where step_1 is the most granular and step_N is the broadest.
    """
    topic_info = topic_model.get_topic_info()
    leaf_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()

    hierarchy_df = topic_model.hierarchical_topics(documents)

    if hierarchy_df.empty:
        # Only one topic (or none) — no merges to process
        result = pd.DataFrame({"Topic": leaf_topics})
        for i in range(1, n_levels + 1):
            result[f"step_{i}"] = result["Topic"].map(
                dict(zip(topic_info["Topic"], topic_info["Name"]))
            )
        return result

    hierarchy_df = hierarchy_df.sort_values("Distance").reset_index(drop=True)
    distances = hierarchy_df["Distance"].values
    thresholds = np.linspace(distances.min(), distances.max(), n_levels + 1)[1:]

    # Build name lookup: leaf topics use their Name from topic_info
    leaf_name_map = dict(zip(topic_info["Topic"], topic_info["Name"]))

    # Pre-parse the merge steps: list of (child_left, child_right, parent_id, parent_name, distance)
    merges = list(
        zip(
            hierarchy_df["Child_Left_ID"],
            hierarchy_df["Child_Right_ID"],
            hierarchy_df["Parent_ID"],
            hierarchy_df["Parent_Name"],
            hierarchy_df["Distance"],
        )
    )

    result = pd.DataFrame({"Topic": leaf_topics})

    for level_i, threshold in enumerate(thresholds, 1):
        # Union-find: map each node to its current group name
        # Start with each leaf mapping to itself
        parent = {t: t for t in leaf_topics}
        group_name = dict(leaf_name_map)

        # Also track merged parent nodes so children of merges can be resolved
        for child_left, child_right, parent_id, parent_name, dist in merges:
            if dist > threshold:
                break
            # Map both children to the parent
            parent[parent_id] = parent_id
            group_name[parent_id] = parent_name

            # Update: point children to parent
            parent[child_left] = parent_id
            parent[child_right] = parent_id

        # Resolve: find root for each leaf topic
        def find_root(node):
            visited = []
            while parent.get(node, node) != node:
                visited.append(node)
                node = parent[node]
            # Path compression
            for v in visited:
                parent[v] = node
            return node

        assignments = {}
        for t in leaf_topics:
            root = find_root(t)
            assignments[t] = group_name[root]

        result[f"step_{level_i}"] = result["Topic"].map(assignments)

    return result


if __name__ == "__main__":
    project_root = Path(find_project_root(__file__))
    data_dir = project_root / "data"
    output_dir = project_root / "output"
    params_path = project_root / "params.yaml"

    full_par = OmegaConf.load(params_path)
    par = full_par.topic_hierarchy
    par_train = full_par.train

    print(f"Starting {Path(__file__).name}")
    start = time.time()
    print_params(full_par)

    # Load documents
    print("Loading data...")
    texts = pl.read_parquet(data_dir / "texts.parquet")
    if par_train.settings.nobs is not None:
        texts = texts.head(par_train.settings.nobs)
    documents = texts["text"].to_list()

    # Load model (use the representation-updated model from evaluate)
    print("Loading model...")
    topic_model = BERTopic.load(output_dir / "bertopic_model_representation")

    # Build hierarchy levels
    print(f"Building hierarchy with {par.n_levels} levels...")
    hierarchy_levels = build_hierarchy_levels(topic_model, documents, par.n_levels)

    # Merge with topic info
    topic_info = topic_model.get_topic_info()
    result = topic_info.merge(hierarchy_levels, on="Topic", how="left")

    # Save
    output_file = output_dir / "topic_hierarchy.csv"
    result.to_csv(output_file, index=False)
    print(f"Saved topic hierarchy to {output_file}")

    hours = (time.time() - start) / 3600
    print(f"Finished {Path(__file__).name} in {hours:.2f} hours")
