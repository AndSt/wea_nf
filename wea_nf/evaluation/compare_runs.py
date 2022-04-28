from typing import List
import os

import json
import pandas as pd

"""
This file wants to load the results of multiple runs.
Ultimately it provides a DataFrame with corresponding statstics.

Typically, we assume the following folder structure:

-   Main dir
-       dataset folder, e.g. "imdb"
-           run folder, "e.g. run_832"
-               contains "config.json" and "{aggregation_scheme}_eval.json" files
"""


def load_json(name: str, dir: str):
    with open(os.path.join(dir, f"{name}.json"), "r") as f:
        return json.load(f)


def get_aggregation_schemes() -> List[str]:
    """The aggregation schemes we want to compare"""
    return ["max", "union", "noisyor"]


def load_iteration(run_dir: str, metric: str = "accuracy"):
    row = {}
    eval_file_names = get_aggregation_schemes()
    eval_files = {}
    for name in eval_file_names:
        eval_files[name] = load_json(f"{name}_eval", run_dir)

    if os.path.isfile(os.path.join(run_dir, "noisyor_eval_2.json")):
        eval_files["simplex"] = load_json(f"noisyor_eval_2", run_dir)

    for name, dict in eval_files.items():
        if metric == "accuracy":
            row[name] = dict.get("accuracy")
        elif metric == "f1":
            row[name] = dict.get("macro avg").get("f1-score")
        else:
            raise ValueError("")
    return row


def load_rows(run_dir: str, metric: str = "accuracy", hyp_params: List[str] = ["lr", "num_epochs"]):
    """Loads results of a single run."""
    row_template = {"run_id": run_dir.split("/")[-1]}
    rows = []

    config = load_json("config", run_dir)
    config_vals = ["dataset", "option", "num_iters", "depth", "weight_decay", "label_dim", "min_matches", "hidden_dim", "batch_size"]
    if "mixing_factor" in config:
        config_vals += ["mixing_factor"]
    for val in config_vals:
        row_template[val] = config.get(val, 0)

    for param in hyp_params:
        row_template[param] = config.get(param)

    iterations = [folder for folder in os.listdir(run_dir) if folder.startswith("iter_")]
    if len(iterations) > 0:
        for iteration in iterations:
            row = load_iteration(run_dir=os.path.join(run_dir, iteration), metric=metric)
            row["iteration"] = iteration.split("_")[1]
            row["iterative"] = int(row["iteration"]) > 0
            row.update(row_template)
            rows.append(row)

    row_template.update(load_iteration(run_dir=run_dir, metric=metric))
    row_template["iteration"] = config.get("num_iters") - 1
    row_template["iterative"] = row_template["iteration"] > 0
    rows += [row_template]

    return rows


def load_dataset_df(run_dir: str, metric: str = "accuracy", hyp_params: List[str] = ["lr", "num_epochs"]):
    """Loads all runs found under run_dir"""
    df = []
    for dir in os.listdir(run_dir):
        if not dir.startswith("run"):
            continue

        dir = os.path.join(run_dir, dir)
        if not os.path.isdir(dir):
            continue
        if "config.json" not in os.listdir(dir):
            continue
        rows = load_rows(dir, metric, hyp_params)
        df += rows

    df = pd.DataFrame(df)
    return df


def filter_and_sort(df: pd.DataFrame, option: str = "multi_rule", eval_type: str = "max"):
    """Filter by option and sort by eval_type"""
    df_filtered = df[df["option"] == option]
    if eval_type == "all":
        df_filtered["all"] = df_filtered[get_aggregation_schemes()].values.max(axis=1)

    df_sorted = df_filtered.sort_values(by=eval_type, ascending=False)
    return df_sorted
