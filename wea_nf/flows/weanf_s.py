from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import torch
from torch import nn

from wea_nf.data.utils import loader_to_np

from wea_nf.layers.real_nvp import RealNVP


class WeaNF_S(nn.Module):
    """
    Standard Model. The corresponding paper calls it WeaNF-I.
    """

    def __init__(
            self,
            input_dim: int = 2, num_classes: int = 2, depth: int = 12,
            label_dim: int = 5, hidden_dim: int = 256,
            T: np.ndarray = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embeddings = torch.nn.Embedding(num_embeddings=num_classes, embedding_dim=label_dim * num_classes)

        self.input_dim = input_dim
        self.flow_dim = self.input_dim + self.embeddings.embedding_dim
        self.flow_hidden_dim = hidden_dim

        self.flow = RealNVP(
            dim=self.input_dim + self.embeddings.embedding_dim, hidden_dim=self.flow_hidden_dim, depth=depth
        )

        if T is not None:
            if T.shape[0] != num_classes:
                raise ValueError("Class Transformation failed")
            self.T = T
        else:
            self.T = np.eye(num_classes)

        self.device = None

    def to_device(self, device):
        if isinstance(device, str):
            device = torch.device(device)

        self.embeddings = self.embeddings.to(device)

        self.flow = self.flow.to(device)
        self.device = device

    def _embed_labels(self, y):
        return self.embeddings(y)

    def log_prob(self, x, y):
        input = torch.cat([x, self._embed_labels(y)], 1)
        return self.flow.log_prob(input)

    def train_loop(
            self,
            data_loader, num_epochs: int = 1, lr: float = 1e-3, x_dev=None, y_dev=None,
            weight_decay: float = 5e-3
    ):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        optimizer = torch.optim.Adam(
            [p for p in self.flow.parameters() if p.requires_grad == True], lr=lr, weight_decay=weight_decay
        )

        self.to_device(device)

        for epoch in tqdm(range(num_epochs)):
            for batch_ndx, (x, y) in enumerate(data_loader):
                self.flow.train()
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                log_prob, log_det = self.log_prob(x, y)
                std_loss = -(log_prob + log_det).mean()

                loss = std_loss
                loss.backward()
                optimizer.step()

            if epoch % 3 == 0:
                self.epoch_stats(data_loader, x_dev, y_dev, epoch, loss)

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

    def predict(self, x, use_T: bool = True, return_proba: bool = False):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.flow.eval()
            x = x.to(device)

            log_probs = []

            for i in range(self.num_classes):
                y = torch.from_numpy(np.ones(x.shape[0]).astype(np.int32) * i).to(device)
                log_prob, log_det = self.log_prob(x, y)
                log_probs.append(log_prob.to(torch.device("cpu")) + log_det.to(torch.device("cpu")))

            full_pred = torch.stack(log_probs, 1)
            y_pred = torch.argmax(full_pred, axis=1).detach().cpu().numpy()

            if return_proba:
                full_pred = torch.softmax(full_pred, dim=1)

            full_pred = full_pred.detach().cpu().numpy()

            if use_T:
                y_pred = self.T.argmax(axis=1)[y_pred.astype(np.int32)]

        return y_pred, full_pred

    def test(self, x, y, use_T: bool = True):
        y_pred, full_pred = self.predict(x, use_T=use_T)
        acc = (pd.Series(y_pred) == pd.Series(y)).mean()
        print(classification_report(y, y_pred))
        return acc

    def epoch_stats(self, data_loader, x_dev, y_dev, epoch, loss):
        if x_dev is not None:
            x, y = loader_to_np(data_loader)
            acc = self.test(x, y, use_T=False)
            print(f"Train accuracy: {acc}", flush=True)

            acc = self.test(x_dev, y_dev, use_T=True)
            print(f"Dev accuracy: {acc}", flush=True)
        print(f"Epoch No. {epoch}, loss: {loss.detach().cpu().numpy()}", flush=True)

    @staticmethod
    def save(save_path: str, net: nn.Module):
        if not save_path.endswith(".pt"):
            raise ValueError("Provide valid file name")

        net.to_device(torch.device("cpu"))
        torch.save(net.state_dict(), save_path)

    @staticmethod
    def load(save_path: str):
        net = WeaNF_S(input_dim=2, num_classes=2)
        net.load_state_dict(torch.load(save_path))
        net.eval()
        return net
