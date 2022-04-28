from typing import Dict, List
import os

import itertools
import json
import joblib


def save(val, name, dir):
    joblib.dump(val, os.path.join(dir, f"{name}.pbz2"))


def json_save(val, name, dir):
    with open(os.path.join(dir, f"{name}.json"), "w") as f:
        json.dump(val, fp=f)


def dict_of_lists_to_list_of_dicts(dict_of_lists: Dict) -> List:
    keys, values = zip(*dict_of_lists.items())
    list_of_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return list_of_dicts
