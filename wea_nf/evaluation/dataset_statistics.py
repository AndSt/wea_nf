from typing import Dict

import numpy as np
import pandas as pd


def get_standard_paper_stats(
        rule_matches_z: np.array, mapping_rules_labels_t: np.array, y_gold: np.ndarray
) -> pd.DataFrame:
    """Computes basic statistics relevant for a paper.
    """
    if mapping_rules_labels_t.shape[1] == 2:
        skewdness = round(y_gold.sum() / y_gold.shape[0], 2)
    else:
        skewdness = None

    stats_dict = [
        ["classes", (y_gold.max() + 1).astype(str)],
        ["train / test samples", f"{rule_matches_z.shape[0]} / {y_gold.shape[0]}"],
        ["rules", rule_matches_z.shape[1]],
        ["avg. rule hits", round(rule_matches_z.sum() / rule_matches_z.shape[0], 2)],
        ["skewdness", skewdness]
    ]
    stats_dict = pd.DataFrame(stats_dict, columns=["statistic", "value"])
    return stats_dict


def combine_multiple_paper_stats(dataset_to_df_dict: Dict) -> pd.DataFrame:
    """Takes a dictionary, having dataset names as keys and DataFrames as values and returns a combined DataFrame.
    """
    columns = []
    values = []

    for dataset, df in dataset_to_df_dict.items():
        columns = df["statistic"].tolist()
        values.append([dataset] + df["value"].tolist())

    stats_df = pd.DataFrame(values, columns=["dataset"] + columns)
    return stats_df