from typing import List, Union
import os
from tqdm.auto import tqdm

import torch
import pandas as pd
import numpy as np

import joblib
from transformers import AutoTokenizer, AutoModel


def convert_text_to_transformer_input(tokenizer, texts: List[str]) -> Union[np.ndarray, np.ndarray]:
    encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding.get('input_ids')
    attention_mask = encoding.get('attention_mask')

    return input_ids, attention_mask


def df_to_transformers_input(
        df: pd.DataFrame, column: str = "review", transformers_model: str = "bert-base-cased"
):
    tokenizer = AutoTokenizer.from_pretrained(transformers_model)
    input_ids, attention_mask = convert_text_to_transformer_input(tokenizer, df[column].tolist())
    return input_ids, attention_mask


def df_to_sentence_rep(
        df: pd.DataFrame, column: str = "review", save_dir: str = None, step_size: int = 100,
        pooling_strategy="mean",
        transformers_model: str = "bert-base-cased"
):
    input_ids, attention_mask = df_to_transformers_input(df, column)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModel.from_pretrained(transformers_model)
    model.to(device)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    full_encoded = []

    with torch.no_grad():
        for i in tqdm(range(0, int(df.shape[0] / step_size) + 1)):

            start_id = i * step_size
            end_id = start_id + step_size

            inp_ids = input_ids[start_id: end_id].to(device)
            att_mask = attention_mask[start_id: end_id].to(device)
            output = model(input_ids=inp_ids, attention_mask=att_mask)

            if pooling_strategy == "mean":
                encoded = output.last_hidden_state.mean(axis=-2)
            elif pooling_strategy == "max":
                encoded = torch.max(output.last_hidden_state, dim=-2).values
            elif pooling_strategy == "natural":
                encoded = output.pooler_output.detach()
            else:
                raise ValueError("Provide correct pooling strategy.")

            encoded = encoded.detach().cpu().numpy()
            if save_dir is not None:
                print(os.path.join(save_dir, f"{start_id}_embs.pbz2"))
                joblib.dump(encoded, os.path.join(save_dir, f"{start_id}_embs.pbz2"))
            else:
                full_encoded.append(encoded)

    if save_dir is not None:
        for i in tqdm(range(0, int(df.shape[0] / step_size) + 1)):
            start_id = i * step_size
            encoded = joblib.load(os.path.join(save_dir, f"{start_id}_embs.pbz2"))
            full_encoded.append(encoded)

    full_encoded = np.vstack(full_encoded)
    return full_encoded
