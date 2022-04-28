from tqdm.auto import tqdm

import numpy as np
from sklearn.metrics import classification_report

import torch
from torch import nn

from wea_nf.layers.real_nvp import RealNVP
from wea_nf.evaluation.multi_lf import compute_noisyor_report

from itertools import chain, combinations


def nonempty_powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    power_chain = chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    return [elt for elt in power_chain if elt != ()]


class WeaNF_M(nn.Module):
    def __init__(
            self,
            input_dim: int = 2, num_classes: int = 2, depth: int = 12,
            label_dim: int = 5, hidden_dim: int = 256,
            T: np.ndarray = None, logger=None
    ):
        super().__init__()
        self.num_classes = num_classes

        self.embedding_dim = label_dim * num_classes
        self.embeddings = torch.nn.Embedding(num_embeddings=num_classes, embedding_dim=self.embedding_dim)

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

        self.logger = logger
        self.device = None

    def log_prob(self, x, y):
        input = torch.cat([x, self.embeddings(y)], 1)
        return self.flow.log_prob(input)

    def to_device(self, device):
        if isinstance(device, str):
            device = torch.device(device)

        self.embeddings = self.embeddings.to(device)

        self.flow = self.flow.to(device)
        self.device = device

    def weight_batch(self, batch_y: torch.Tensor, random=True):
        """Returns weighting array alpha; forms convex sum.

        shape: (batch_dim, num_rules)
        """
        batch_y = batch_y.float()
        batch_y += 0.1 * torch.ones(batch_y.shape)
        if random == True:
            batch_y = batch_y * torch.rand(batch_y.shape)
        nbatch_y = batch_y / batch_y.sum(axis=1, keepdims=True)
        return nbatch_y

    def mix_and_embed(self, x, y, z, random=True):
        mixed_y = self.weight_batch(y.to(torch.device("cpu")), random=random)
        x, mixed_y = x.to(self.device), mixed_y.to(self.device)
        embs = torch.matmul(mixed_y, self.embeddings.weight)
        embs = torch.cat([x, embs], dim=1)
        return embs

    def train_loop(
            self,
            data_loader, num_epochs: int = 1, lr: float = 1e-3, x_dev=None, y_dev=None,
            weight_decay: float = 5e-3, balance_classes: bool = True
    ):
        parameters = [p for p in self.flow.parameters() if p.requires_grad == True]
        parameters += [p for p in self.embeddings.parameters() if p.requires_grad == True]
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to_device(device)

        for epoch in tqdm(range(num_epochs)):
            for batch_ndx, (x, y, z) in enumerate(data_loader):
                self.flow.train()
                optimizer.zero_grad()

                embs = self.mix_and_embed(x, y, z)

                log_prob, log_det = self.flow.log_prob(embs)
                std_loss = -(log_prob + log_det).mean()

                loss = std_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

            if epoch % 3 == 0:
                self.epoch_stats(x_dev, y_dev, epoch, loss)

    def epoch_stats(self, x_dev, y_dev, epoch, loss):
        if x_dev is not None:
            eval_dict = self.test(x_dev, y_dev, use_T=True)
            acc = eval_dict.get("accuracy")
            f1 = eval_dict.get("macro avg", {}).get("f1-score")
            print(f"Dev accuracy: {acc}", flush=True)
            if self.logger is not None:
                self.logger.update(
                    clock_tick={"num_epochs": epoch},
                    stats_tick={
                        "train_loss": loss.detach().cpu().numpy(),
                        'test_accuracy': acc,
                        "test_f1": f1
                    },
                    save=True
                )
        print(f"Epoch No. {epoch}, loss: {loss.detach().cpu().numpy()}", flush=True)

    def simplex_predict(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.flow.eval()
            x = x.to(device)
            log_probs_full = []
            for j in range(15):
                log_probs = []
                for i in range(self.T.shape[1]):
                    class_emb = np.repeat(np.expand_dims(self.T[:, i], axis=0), x.shape[0], axis=0)
                    class_emb = torch.from_numpy(class_emb).float().to(device)
                    class_emb = self.mix_and_embed(x, class_emb, None)
                    log_prob, log_det = self.flow.log_prob(class_emb)
                    log_probs.append(log_prob.to(torch.device("cpu")) + log_det.to(torch.device("cpu")))
                full_pred = torch.stack(log_probs, 1).detach().cpu()  # .numpy()
                log_probs_full.append(full_pred)

        return log_probs_full

    def full_predict(self, x, match_sets=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if match_sets is None:
            match_sets = {}
            for i in list(range(x.shape[0])):
                match_sets[i] = nonempty_powerset(list(range(self.num_classes)))

        input, input_combinations = self._embed(x, match_sets=match_sets)
        log_prob, log_det = self.flow.log_prob(input)
        return log_prob + log_det, input_combinations

    def predict(self, x, use_T: bool = True):
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

            full_pred_1 = torch.stack(log_probs, 1)
            y_pred_1 = torch.argmax(full_pred_1, axis=1).detach().cpu().numpy()

            full_pred_1 = full_pred_1.detach().cpu().numpy()

            if use_T:
                y_pred_1 = self.T.argmax(axis=1)[y_pred_1.astype(np.int32)]

            if use_T:
                log_probs = []
                for i in range(self.T.shape[1]):
                    class_emb = np.repeat(np.expand_dims(self.T[:, i], axis=0), x.shape[0], axis=0)
                    class_emb = torch.from_numpy(class_emb).float().to(device)
                    class_emb = self.mix_and_embed(x, class_emb, None, random=False)
                    log_prob, log_det = self.flow.log_prob(class_emb)
                    log_probs.append(log_prob.to(torch.device("cpu")) + log_det.to(torch.device("cpu")))
                full_pred_2 = torch.stack(log_probs, 1).detach().cpu().numpy()
            else:
                full_pred_2 = None

        return y_pred_1, full_pred_1, full_pred_2

    def test(self, x, y, use_T: bool = True):
        y_pred, full_pred, full_pred_2 = self.predict(x, use_T=use_T)

        print("standard")
        print(classification_report(y, y_pred))
        print("noisyor")
        print(compute_noisyor_report(y, full_pred, self.T, y_train=None))
        print("integral")
        print(classification_report(y, full_pred_2.argmax(axis=1)))
        print("simplex")
        full_pred_simplex = self.simplex_predict(x)
        y_pred = torch.logsumexp(torch.stack(full_pred_simplex, 2), 2).numpy()
        print(classification_report(y, y_pred.argmax(axis=1)))
        return compute_noisyor_report(y, full_pred, self.T, y_train=None)

    @staticmethod
    def save(save_path: str, net: nn.Module):
        if not save_path.endswith(".pt"):
            raise ValueError("Provide valid file name")

        net.to(torch.device("cpu"))
        torch.save(net.state_dict(), save_path)

    @staticmethod
    def load(save_path: str):
        net = WeaNF_M(input_dim=2, num_classes=2)
        net.load_state_dict(torch.load(save_path))
        net.eval()
        return net
