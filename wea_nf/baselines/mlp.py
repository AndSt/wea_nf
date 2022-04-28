import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 768, num_classes: int = 2, num_layers: int = 3, hidden_dim: int = 768):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for i in range(num_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ]
            if i % 2 == 0:
                layers += [nn.Dropout(0.5)]

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def train(
        loader, num_epochs: int = 10, num_classes: int = 2, num_layers: int = 6, lr: float = 1e-3,
        dev_loader=None, test_loader=None
):
    net = SimpleMLP(num_classes=num_classes, num_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            net.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()

        train_acc, _, _ = predict(net, loader)

        if dev_loader is not None:
            dev_acc, _, _ = predict(net, dev_loader)
            print(f"Epoch {epoch}: Loss {running_loss}, Train accuracy: {train_acc}, Dev accuracy {dev_acc}")
    _, y_test_pred, y_test_truth = predict(net, test_loader)

    return evaluate(y_true=y_test_truth, y_pred=y_test_pred)


def predict(model, loader):
    model.eval()

    predictions = []
    correct = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            input, labels = data
            # calculate outputs by running images through the network
            outputs = model(input)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted
            correct += list(labels.numpy())
            predictions += list(predicted.numpy())

    acc = (pd.Series(predictions) == pd.Series(correct)).mean()
    return acc, predictions, correct


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1
