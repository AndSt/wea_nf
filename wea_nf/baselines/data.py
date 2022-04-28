import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy.sparse as sp

from snorkel.labeling.model import LabelModel

from wea_nf.data.knodle_format import get_mixing_data, get_mv_data


def z_t_matrix_to_snorkel_matrix(rule_matches_z: np.ndarray, mapping_rules_labels_t: np.ndarray) -> np.ndarray:
    """Transforms Z, T matrices to the Snorkel format. More specifically, the Snorkel matrix has the same shape as Z,
    but either holds -1 for no match or the label index. Thus the additional assumption is made that one LF is able
    to label multiple labelling functions.
    """
    snorkel_matrix = -1 * np.ones(rule_matches_z.shape)

    if isinstance(rule_matches_z, sp.csr_matrix):
        rule_matches_z = rule_matches_z.toarray()

    t_to_label_id = np.argmax(mapping_rules_labels_t, axis=-1)

    if isinstance(mapping_rules_labels_t, sp.csr_matrix):
        # transform np.matrix to np.array
        t_to_label_id = np.array(t_to_label_id).flatten()

    for i in range(rule_matches_z.shape[0]):
        non_zero_idx = np.nonzero(rule_matches_z[i])[0]
        snorkel_matrix[i, non_zero_idx] = t_to_label_id[non_zero_idx]

    return snorkel_matrix


def mv_data(data_dir, min_matches=30, batch_size=256):
    (
        X_train, y_train_mv, X_unlabelled, T, (y_train_mv_true, y_train_true),
        X_dev, y_dev,
        X_test, y_test
    ) = get_mv_data(data_dir, min_matches=min_matches)

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train_mv.astype(np.long))
    )
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=1)

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=1)

    return loader, test_loader, T.shape[1]


def snorkel_data(data_dir, min_matches=30, batch_size=256):
    (
        X_train, y_train, X_unlabelled, T, (Z_labelled, Z_unlabelled),
        X_dev, y_dev,
        X_test, y_test
    ) = get_mixing_data(data_dir, min_matches=min_matches)

    L_train = z_t_matrix_to_snorkel_matrix(Z_labelled, T)

    label_model = LabelModel(cardinality=T.shape[1], verbose=True)
    label_model.fit(
        L_train,
        n_epochs=5000,
        log_freq=500,
    )

    y_train_snorkel = label_model.predict_proba(L_train).argmax(axis=1)

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train_snorkel.astype(np.long))
    )
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=1)

    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, num_workers=1)

    return loader, test_loader, T.shape[1]
