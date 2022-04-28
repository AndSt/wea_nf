import os
from tqdm.auto import tqdm

import torch
import pandas as pd
import numpy as np
import joblib

from sentence_transformers import SentenceTransformer


def df_to_sentence_rep(
        df: pd.DataFrame, column: str = "review", save_dir: str = None, step_size: int = 1000,
        sentence_bert_version: str = 'bert-base-nli-mean-tokens'
):
    model = SentenceTransformer(sentence_bert_version)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    full_encoded = []
    with torch.no_grad():
        for i in tqdm(range(0, int(df.shape[0] / step_size) + 1)):
            start_id = i * step_size
            end_id = start_id + step_size
            inp = df[start_id: end_id][column].tolist()
            encoded = model.encode(inp)
            if save_dir is not None:
                joblib.dump(encoded, os.path.join(save_dir, f"{start_id}_embs.pbz2"))
            elif encoded.shape[0] > 0:
                full_encoded.append(encoded)

    if save_dir is not None:
        for i in tqdm(range(0, int(df.shape[0] / step_size) + 1)):
            start_id = i * step_size
            encoded = joblib.load(os.path.join(save_dir, f"{start_id}_embs.pbz2"))
            if encoded.shape[0] > 0:
                full_encoded.append(encoded)

    full_encoded = np.vstack(full_encoded)
    return full_encoded
