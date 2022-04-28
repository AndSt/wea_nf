import os
from typing import Dict, List

import itertools
import argparse
import json
import random

import torch
import pandas as pd
import numpy as np
import joblib
from mle_logging import MLELogger

from torch.utils.data import DataLoader, TensorDataset

from wea_nf.data.knodle_format import get_weak_data
from wea_nf.data.imbalanced_sampler import ImbalancedDatasetSampler

from wea_nf.flows.weanf_n import WeaNF_N

from wea_nf.experiments import json_save, save, dict_of_lists_to_list_of_dicts
from wea_nf.evaluation.multi_lf import compute_max, compute_union, compute_noisyor_report


def convert_param_dict(dict: Dict):
    for key, val in dict.items():
        if key in ["lr", "weight_decay"]:
            dict[key] = float(val)
        elif key in [
            "num_epochs", "batch_size", "label_dim", "hidden_dim", "mixing_factor",
            "depth", "min_matches"
        ]:
            dict[key] = int(val)
    return dict


def run(
        data_dir: str, save_dir: str,
        dataset: str, emb_type: str, option: str, min_matches: int,
        depth: int, label_dim: int, hidden_dim: int, mixing_factor: int,
        lr: float, batch_size: int, weight_decay: float, num_epochs: int
):
    (
        X_train, y_train, X_unlabelled, T, (Z_labelled, Z_unlabelled),
        X_dev, y_dev,
        X_test, y_test
    ) = get_weak_data(data_dir, min_matches=min_matches)

    logger = MLELogger(
        time_to_track=['num_epochs'],
        what_to_track=['train_loss', 'train_accuracy', 'test_accuracy'],
        experiment_dir=f"{save_dir}",
        model_type='torch'
    )
    tensor_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.int32)),
        torch.from_numpy(Z_labelled.astype(np.int32))
    )
    sampler = ImbalancedDatasetSampler(dataset=tensor_dataset, T=T)
    loader = DataLoader(dataset=tensor_dataset, batch_size=batch_size, num_workers=1, sampler=sampler)

    c = WeaNF_N(
        input_dim=X_train.shape[1], num_classes=Z_labelled.shape[1], depth=depth,
        label_dim=label_dim, hidden_dim=hidden_dim, mixing_factor=mixing_factor,
        T=T
    )

    c.train_loop(
        loader, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, x_dev=X_test, y_dev=y_test, logger=logger
    )

    config = {
        "data_dir": data_dir,
        "save_dir": save_dir,
        "dataset": dataset,
        "emb_type": emb_type,
        "min_matches": min_matches,
        "depth": depth,
        "label_dim": label_dim,
        "hidden_dim": hidden_dim,
        "mixing_factor": mixing_factor,
        "option": option,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs
    }
    json_save(config, "config", save_dir)

    y_pred, full_pred, _, _ = c.predict(X_test)
    print("test acc:", (pd.Series(y_pred) == pd.Series(y_test)).mean())

    if X_dev is not None:
        y_pred_dev, full_pred_dev, _, _ = c.predict(X_dev, use_T=False)
        save(X_dev, "x_dev", save_dir)
        save(y_dev, "y_dev", save_dir)
        save(y_pred_dev, "y_pred_dev", save_dir)
        save(full_pred_dev, "y_full_pred_dev", save_dir)

    y_pred_test, full_pred_test, pos_log_probs, neg_log_probs = c.predict(X_test, use_T=False)
    save(X_test, "x_test", save_dir)
    save(y_test, "y_test", save_dir)
    save(y_pred_test, "y_pred_test", save_dir)
    save(full_pred_test, "y_full_pred_test", save_dir)

    json_save(compute_max(y_test, full_pred, T, y_train=None), "max_eval", save_dir)
    json_save(compute_union(y_test, full_pred, T, y_train=None), "union_eval", save_dir)
    json_save(compute_noisyor_report(y_test, full_pred, T, y_train=None), "noisyor_eval", save_dir)


if __name__ == '__main__':
    # actual run
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)

    parser.add_argument('--dataset', default="imdb")
    parser.add_argument('--emb_type', default="sentence")
    parser.add_argument('--option', default="multi_rule")
    parser.add_argument('--min_matches', default=200)

    parser.add_argument('--depth', default=12)
    parser.add_argument('--label_dim', default=5)
    parser.add_argument('--hidden_dim', default=256)
    parser.add_argument('--mixing_factor', default=3)

    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--weight_decay', default=5e-2)
    parser.add_argument('--num_epochs', default=80)

    args = parser.parse_args()

    hyps = [
        "dataset", "emb_type", "option", "min_matches",
        "depth", "label_dim", "hidden_dim", "mixing_factor",
        "lr", "weight_decay", "num_epochs", "batch_size"
    ]
    arg_dict = {}

    for hyp in hyps:
        arg = args.__dict__[hyp]
        if isinstance(arg, str):
            arg_dict[hyp] = arg.split(",")
        else:
            arg_dict[hyp] = [arg]

    params = dict_of_lists_to_list_of_dicts(arg_dict)
    for param in params:
        param = convert_param_dict(param)

        data_dir = os.path.join(args.data_dir, param['dataset'], "processed", param['emb_type'])
        save_dir = os.path.join(args.save_dir, param['dataset'], f"run_{random.randint(0, 1000)}")
        while os.path.isdir(save_dir):
            save_dir = os.path.join(args.save_dir, param['dataset'], f"run_{random.randint(0, 1000)}")
        os.makedirs(save_dir, exist_ok=True)

        print(f"Run: {param}")
        print(f"Data_dir: {data_dir}")
        print(f"Save_dir: {save_dir}")
        run(data_dir=data_dir, save_dir=save_dir, **param)
        print(" ")
        print(" ")

    # run(data_dir=args.data_dir, save_dir=args.save_dir)
