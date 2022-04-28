from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import random
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import torch
from torch import nn

from wea_nf.layers.real_nvp import RealNVP
from wea_nf.evaluation.multi_lf import compute_noisyor_report


def matches_to_match_lists(z: np.ndarray) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Returns the list of matched samples for each LF"""
    matched_indices, non_matched_indices = {}, {}
    for i in range(z.shape[1]):
        matched_indices[i] = []
        non_matched_indices[i] = []

    indices = np.where(z == 1)
    for i, j in zip(*indices):
        matched_indices[j].append(i)

    indices = np.where(z == 0)
    for i, j in zip(*indices):
        non_matched_indices[j].append(i)

    return matched_indices, non_matched_indices


def lf_z_to_pairs(lf: torch.Tensor, z: torch.Tensor, negative_samples: int = 1):
    if not isinstance(z, np.ndarray):
        lf = lf.cpu().detach().numpy()
        z = z.cpu().detach().numpy()

    pos_pairs = np.concatenate([np.arange(lf.shape[0]), lf]).reshape((2, -1)).T

    matched_indices, non_matched_indices = matches_to_match_lists(z)
    neg_pairs = []
    for i in range(pos_pairs.shape[0]):
        try:
            sample_list = non_matched_indices[pos_pairs[i, 1]]
            neg_idx = random.sample(sample_list, min(len(sample_list), negative_samples))
        except:
            print(pos_pairs[i, 1])
            print(non_matched_indices)
            print(lf.shape, z.shape)
            assert 0 == 1
        for j in neg_idx:
            neg_pairs.append([j, pos_pairs[i, 1]])
    neg_pairs = np.array(neg_pairs)

    return pos_pairs, neg_pairs


class WeaNF_N(nn.Module):
    def __init__(
            self,
            input_dim: int = 2, num_classes: int = 2, depth: int = 12,
            label_dim: int = 5, hidden_dim: int = 256, mixing_factor: int = 3,
            T: np.ndarray = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mixing_factor = mixing_factor

        self.embedding_dim = label_dim * num_classes
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=num_classes, embedding_dim=self.embedding_dim)
        self.neg_embeddings = torch.nn.Embedding(num_embeddings=num_classes, embedding_dim=self.embedding_dim)

        self.input_dim = input_dim
        self.flow_dim = self.input_dim + self.embedding_dim
        self.flow_hidden_dim = hidden_dim

        self.flow = RealNVP(
            dim=self.input_dim + self.embedding_dim, hidden_dim=self.flow_hidden_dim, depth=depth
        )

        if T is not None:
            if T.shape[0] != num_classes:
                raise ValueError("Class Transformation failed")
            self.T = T
        else:
            self.T = np.eye(num_classes)

        self.device = None

    def log_prob(self, x, y):
        emb = self.pos_embeddings(y)
        input = torch.cat([x, emb], 1)
        neg_input = torch.cat([x, self.neg_embeddings(y)], 1)
        return self.flow.log_prob(input), self.flow.log_prob(neg_input)

    def to_device(self, device):
        if isinstance(device, str):
            device = torch.device(device)

        self.pos_embeddings = self.pos_embeddings.to(device)
        self.neg_embeddings = self.neg_embeddings.to(device)

        self.flow = self.flow.to(device)
        self.device = device

    def _embed(self, x, pairs: torch.Tensor, emb_type: str = "positive"):

        x = x.to(self.device)
        pairs = pairs.to(self.device)
        x_emb = x[pairs[:, 0]]

        if emb_type == "positive":
            pos = self.pos_embeddings(pairs[:, 1])
            input = torch.cat([x_emb, pos], dim=1)
        elif emb_type == "negative":
            neg = self.neg_embeddings(pairs[:, 1])
            input = torch.cat([x_emb, neg], dim=1)
        else:
            raise ValueError("Provide valid embedding type.")

        return input

    def mix_and_embed(self, x, lf, z):
        pos_pairs, neg_pairs = lf_z_to_pairs(lf, z, negative_samples=self.mixing_factor)

        pos_pairs = torch.from_numpy(pos_pairs).to(self.device)
        neg_pairs = torch.from_numpy(neg_pairs).to(self.device)
        pos_embs = self._embed(x, pos_pairs, "positive")
        neg_embs = self._embed(x, neg_pairs, "negative")
        return pos_embs, neg_embs

    def train_loop(
            self,
            data_loader, num_epochs: int = 1, lr: float = 1e-3, x_dev=None, y_dev=None,
            weight_decay: float = 5e-3, logger=None
    ):

        optimizer = torch.optim.Adam(
            [p for p in self.flow.parameters() if p.requires_grad == True], lr=lr, weight_decay=weight_decay
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to_device(device)

        for epoch in tqdm(range(num_epochs)):
            for batch_ndx, (x, lf, z) in enumerate(data_loader):
                self.flow.train()
                optimizer.zero_grad()

                pos_embs, neg_embs = self.mix_and_embed(x, lf, z)

                pos_log_prob, pos_log_det = self.flow.log_prob(pos_embs)
                neg_log_prob, neg_log_det = self.flow.log_prob(neg_embs)

                std_loss = -(pos_log_prob + pos_log_det).mean() - (neg_log_prob + neg_log_det).mean()

                loss = std_loss
                loss.backward()
                optimizer.step()

            if epoch % 3 == 0:
                self.epoch_stats(x_dev, y_dev, epoch, loss, logger=logger)

    def embed(self, x, y):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y.astype(np.int32))
        x = x.to(device)
        y = y.to(device)
        inp = torch.cat([x, self._embed_labels(y)], 1).to(device)

        z, _ = self.flow.forward(inp)
        z = z.detach().cpu().numpy()
        return z

    def epoch_stats(self, x_dev, y_dev, epoch, loss, logger=None):
        if x_dev is not None:
            acc, f1 = self.test(x_dev, y_dev, use_T=True, return_noisyor=True)
            print(f"Dev accuracy: {acc}", flush=True)
            if logger is not None:
                logger.update(
                    clock_tick={"num_epochs": epoch},
                    stats_tick={
                        "train_loss": loss.detach().cpu().numpy(),
                        'test_accuracy': acc,
                        "test_f1": f1
                    },
                    save=True
                )

        print(f"Epoch No. {epoch}, loss: {loss.detach().cpu().numpy()}", flush=True)

    def predict(self, x, use_T: bool = True):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.flow.eval()
            x = x.to(device)

            pos_log_probs = []
            neg_log_probs = []

            for i in range(self.num_classes):
                y = torch.from_numpy(np.ones(x.shape[0]).astype(np.int32) * i).to(device)
                (pos_log_prob, pos_log_det), (neg_log_prob, neg_log_det) = self.log_prob(x, y)
                pos_log_probs.append(pos_log_prob.to(torch.device("cpu")) + pos_log_det.to(torch.device("cpu")))
                neg_log_probs.append(neg_log_prob.to(torch.device("cpu")) + neg_log_det.to(torch.device("cpu")))

            pos_log_probs = torch.stack(pos_log_probs, 1)
            neg_log_probs = torch.stack(neg_log_probs, 1)
            stacked = torch.stack([pos_log_probs, neg_log_probs])

            normalizer = torch.logsumexp(stacked, dim=0)
            full_pred = pos_log_probs - normalizer

            y_pred = torch.argmax(full_pred, axis=1).detach().cpu().numpy()

            full_pred = full_pred.detach().cpu().numpy()

            if use_T:
                y_pred = self.T.argmax(axis=1)[y_pred.astype(np.int32)]

        return y_pred, full_pred, pos_log_probs.detach().cpu(), neg_log_probs.detach().cpu()

    def test(self, x, y, use_T: bool = True, return_noisyor=False):
        y_pred, full_pred, pos_log_probs, neg_log_probs = self.predict(x, use_T=use_T)
        acc = (pd.Series(y_pred) == pd.Series(y)).mean()
        print(classification_report(y, y_pred))
        y_pred_old = self.T.argmax(axis=1)[torch.argmax(pos_log_probs, axis=1).detach().cpu().numpy().astype(np.int32)]
        print(classification_report(y, y_pred_old))
        noisyor = compute_noisyor_report(y, full_pred, self.T, y_train=None)
        print(noisyor)
        if return_noisyor:
            return noisyor.get("accuracy"), noisyor.get("macro avg").get("f1-score")
        return acc

    @staticmethod
    def save(save_path: str, net: nn.Module):
        if not save_path.endswith(".pt"):
            raise ValueError("Provide valid file name")

        net.to(torch.device("cpu"))
        torch.save(net.state_dict(), save_path)

    @staticmethod
    def load(save_path: str):
        net = WeaNF_N(input_dim=2, num_classes=2)
        net.load_state_dict(torch.load(save_path))
        net.eval()
        return net
