from typing import Dict, List
import os
import itertools

from tqdm import tqdm

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from wea_nf.data.knodle_format import get_unprocessed_dir, get_samples, filter_small_lfs
from wea_nf.baselines.data import mv_data, snorkel_data
import wea_nf.baselines.mlp as mlp


def dict_of_lists_to_list_of_dicts(dict_of_lists: Dict) -> List:
    keys, values = zip(*dict_of_lists.items())
    list_of_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return list_of_dicts


def evaluate_pure_mv(data_dir: str, min_matches: int = None):
    unprocessed_dir = get_unprocessed_dir(data_dir=data_dir)

    T = joblib.load(os.path.join(unprocessed_dir, "mapping_rules_labels_t.lib"))
    Z_orig = joblib.load(os.path.join(data_dir, "test_rule_matches_z.pbz2"))
    _, y_true = get_samples(data_dir, split="test")

    if isinstance(min_matches, int):
        filtered_lfs, Z, T = filter_small_lfs(Z_orig, T, min_matches=min_matches)
    else:
        Z = Z_orig

    y = np.dot(Z, T)

    y_pred = []
    for i in range(y.shape[0]):
        row_max = np.max(y[i])
        num_occurrences = (row_max == y[i]).sum()
        if num_occurrences != 1:
            max_ids = np.where(y[i] == row_max)[0]
            y_pred.append(int(np.random.choice(max_ids)))
        else:
            y_pred.append(np.argmax(y[i]))

    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1


def train_mv_baseline(base_dir: str, datasets: List[str] = None):
    datasets_dict = {
        "imdb": "accuracy",
        "spouse": "f1",
        "spam": "accuracy",
        "sms": "f1",
        "trec": "accuracy"
    }
    if datasets is not None:
        datasets_dict = {k: v for k, v in datasets_dict.items() if k in datasets}
    else:
        datasets = list(datasets_dict.keys())

    row = {"method": "Pure MV"}
    for dataset in tqdm(datasets_dict):
        data_dir = os.path.join(base_dir, dataset, "processed", "sentence")

        accs, f1s = [], []
        for i in range(200):
            acc, f1 = evaluate_pure_mv(data_dir)
            accs.append(acc)
            f1s.append(f1)

        if datasets_dict[dataset] == "accuracy":
            row[dataset] = sum(accs) / len(accs)
        else:
            row[dataset] = sum(f1s) / len(f1s)

    row_df = pd.DataFrame([row])
    row_df[datasets] = row_df[datasets] * 100
    row_df = row_df.round(2)
    return row_df


def transform_trainable_df(df: pd.DataFrame, datasets: List[str]) -> pd.DataFrame:
    datasets_dict = {
        "imdb": "accuracy",
        "spouse": "f1",
        "spam": "accuracy",
        "sms": "f1",
        "trec": "accuracy"
    }
    if datasets is not None:
        datasets_dict = {k: v for k, v in datasets_dict.items() if k in datasets}
    else:
        datasets = list(datasets_dict.keys())

    result_df = []

    for method in ["MV", "Snorkel"]:
        row = {}
        row["method"] = method
        for dataset in datasets_dict:
            result_row = df[df["dataset"] == dataset]
            result_row = result_row[result_row["type"] == method]
            result_row = result_row.sort_values(by=datasets_dict[dataset], ascending=False)
            row[dataset] = result_row.iloc[0][datasets_dict[dataset]]
        result_df.append(row)

    result_df = pd.DataFrame(result_df)
    result_df[datasets] = result_df[datasets] * 100
    result_df = result_df.round(2)
    return result_df


def train_trainable_baselines(base_dir: str = None, datasets: List[str] = None, hyps: Dict = None):
    if datasets is None:
        datasets = ["spouse", "spam", "imdb", "trec", "sms"]
    if hyps is None:
        hyps = {
            "batch_size": [256],
            "min_matches": [0],  # 30, 100, 150],
            "num_epochs": [1],
            "num_layers": [3],
            "lr": [1e-2, 1e-3]
        }

    hyps = dict_of_lists_to_list_of_dicts(hyps)

    df = []

    for dataset in datasets:
        data_dir = os.path.join(base_dir, dataset, "processed", "sentence")

        # majority vote
        for hyp in tqdm(hyps):
            if dataset == "sms" and hyp.get("min_matches") > 100:
                continue

            loader, test_loader, num_classes = mv_data(
                data_dir,
                min_matches=hyp.get("min_matches"),
                batch_size=hyp.get("batch_size")
            )
            print(dataset, len(loader), hyp)
            accs = []
            f1s = []
            for i in range(5):
                acc, f1 = mlp.train(
                    loader, num_classes=num_classes, num_layers=hyp.get("num_layers"),
                    lr=hyp.get("lr"), num_epochs=hyp.get("num_epochs"),
                    dev_loader=test_loader, test_loader=test_loader
                )
                accs.append(acc)
                f1s.append(f1)
                print(" ")

            arr = [
                "MV", dataset, sum(accs) / len(accs), sum(f1s) / len(f1s),
                hyp.get("num_epochs"), hyp, {"accs": accs, "f1s": f1s}
            ]
            df.append(arr)

            # snorkel
            loader, test_loader, num_classes = snorkel_data(
                data_dir,
                min_matches=hyp.get("min_matches"),
                batch_size=hyp.get("batch_size")
            )
            accs = []
            f1s = []
            for i in range(5):
                acc, f1 = mlp.train(
                    loader, num_classes=num_classes, num_layers=hyp.get("num_layers"),
                    lr=hyp.get("lr"), num_epochs=hyp.get("num_epochs"),
                    dev_loader=test_loader, test_loader=test_loader
                )
                accs.append(acc)
                f1s.append(f1)
                print(" ")

            arr = [
                "Snorkel", dataset, sum(accs) / len(accs), sum(f1s) / len(f1s),
                hyp.get("num_epochs"), hyp, {"accs": accs, "f1s": f1s}
            ]
            df.append(arr)

    df = pd.DataFrame(df, columns=["type", "dataset", "accuracy", "f1", "num_epochs", "hyper_params", "numbers"])
    df = transform_trainable_df(df, datasets)
    return df


def full_baseline_df(data_dir: str, datasets):
    df_mv = train_mv_baseline(data_dir, datasets)
    df_trainable = train_trainable_baselines(data_dir, datasets)

    baseline_df = pd.concat([df_mv, df_trainable])
    return baseline_df