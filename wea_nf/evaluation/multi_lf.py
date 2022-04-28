import pandas as pd
import numpy as np
import torch
import random

from sklearn.metrics import classification_report

"""Multi Labelling function module

Within this file we contain code to aggregate multiple labelling functions.
Thus the code here is used after the normalizing flow is trained.
"""


def get_p_lambda(y_train):
    # basic statistics:
    p_lambda = pd.Series(y_train).value_counts().sort_index()
    p_lambda = (p_lambda / p_lambda.sum()).values
    return p_lambda


def compute_max(y_test, y_pred_log_prob, T, y_train):
    y_pred = np.argmax(T, axis=1)[y_pred_log_prob.argmax(axis=1)]
    return classification_report(y_test, y_pred, digits=4, output_dict=True)


def get_x_lambda_softmax(y_pred_log_prob):
    y_pred_log_prob = torch.from_numpy(y_pred_log_prob)
    y_pred = torch.softmax(y_pred_log_prob, dim=1).numpy()
    return y_pred


def compute_union(y_test, y_pred_log_prob, T, y_train):
    y_pred = get_x_lambda_softmax(y_pred_log_prob)
    y_pred = np.dot(y_pred, T)
    return classification_report(y_test, y_pred.argmax(axis=1), digits=4, output_dict=True)


def get_p_lambda_count(T, y_train):
    a = pd.Series(y_train).value_counts().reset_index()
    a.columns = ["lf", "num_occurences"]

    p_lambda_count = np.zeros(T.shape[0])
    for idx, row in a.iterrows():
        p_lambda_count[row["lf"]] = row["num_occurences"]
    return p_lambda_count

#
# def compute_inversion(y_test, y_log_prob, T, y_train):
#     p_lambda_count = get_p_lambda_count(T, y_train)
#     c_count = pd.Series(T[y_train].argmax(axis=1)).value_counts().sort_index().values
#
#     p_lambda_c_unsup = T * np.expand_dims(p_lambda_count, axis=1)
#     p_lambda_c_unsup = p_lambda_c_unsup / c_count
#
#     shape = y_log_prob.shape
#     num_classes = len(c_count)
#     ext_log_prob = np.repeat(y_log_prob, num_classes, axis=1).reshape((shape[0], num_classes, shape[1]))
#     ext_log_prob += np.log(p_lambda_c_unsup.T)
#
#     p_x_given_c = torch.zeros((shape[0], num_classes))
#     for c in range(num_classes):
#         # get indices of lambdas
#         idx = np.where(T[:, c] != 0)[0]
#         subset = ext_log_prob[:, c, idx]
#         p_x_given_c[:, c] = torch.logsumexp(torch.from_numpy(subset), dim=1)
#     # p_x_given_c += np.log(np.array([0.8, 0.2]))
#     return classification_report(y_test, np.argmax(p_x_given_c, axis=1), digits=4, output_dict=True)


def compute_noisyor(y_full_pred_test, T):
    y_pred_local = get_x_lambda_softmax(y_full_pred_test)
    p = np.zeros((y_pred_local.shape[0], T.shape[1]))

    for i in range(T.shape[1]):
        idx = np.where(T[:, i] == 1)[0]

        if len(idx) == 0:
            p[:, i] = 1
        else:
            p[:, i] = np.exp((np.log((1 - y_pred_local[:, idx])).mean(axis=1)))
    y_pred_noisyor = np.argmax(1 - p, axis=1)
    return y_pred_noisyor


def compute_negative_noisyor(y_full_pred_test, T):
    p = np.zeros((y_full_pred_test.shape[0], T.shape[1]))

    for i in range(T.shape[1]):
        idx = np.where(T[:, i] == 1)[0]

        if len(idx) == 0:
            p[:, i] = 1
        else:
            p[:, i] = np.exp((np.log((1 - y_full_pred_test[:, idx])).mean(axis=1)))
    y_pred_noisyor = np.argmax(1 - p, axis=1)
    return y_pred_noisyor


def compute_noisyor_report(y_test, y_full_pred_test, T, y_train):
    y_pred_noisyor = compute_noisyor(y_full_pred_test, T)
    return classification_report(y_test, y_pred_noisyor, digits=4, output_dict=True)


def proba_mv(y_test, y_full_pred_test, T, y_train, y_full_pred_train=None):
    lambda_counts = get_p_lambda_count(T, y_train)
    lambda_counts = lambda_counts / lambda_counts.sum()
    thresholds = []

    full_pred = y_full_pred_test
    for i in range(full_pred.shape[1]):
        a = np.sort(full_pred[:, i])[::-1]
        index = int(lambda_counts[i] * full_pred.shape[0])  # int(lambda_counts[i] / y_test.shape[0])
        thresholds.append(a[index])
    thresholds = np.array(thresholds)
    # y_pred, full_pred = c.predict(X_test, use_T=False)
    votes = np.dot((y_full_pred_test > thresholds), T)

    y_test_mv = []
    for x in votes:
        if x[0] == x[1]:
            y_test_mv.append(random.randint(0, 1))
            # y_test_mv.append(-1)
        else:
            y_test_mv.append(np.argmax(x))
    y_test_mv = np.array(y_test_mv)

    return classification_report(y_test[y_test_mv != -1], y_test_mv[y_test_mv != -1], digits=4, output_dict=True)
