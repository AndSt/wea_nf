import numpy as np
from wea_nf.flows.weanf_m import matches_to_match_lists


def test_matches_to_match_list():
    z = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])

    test_matched_indices = {0: [0, 1], 1: [1, 2], 2: [2]}
    test_non_matched_indices = {0: [2], 1: [0], 2: [0, 1]}

    matched_indices, non_matched_indices = matches_to_match_lists(z)
    for i in test_matched_indices.keys():
        assert test_matched_indices[i] == matched_indices[i]
        assert test_non_matched_indices[i] == non_matched_indices[i]
