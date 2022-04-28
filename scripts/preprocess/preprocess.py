import os
import argparse

import joblib
import pandas as pd

import trafos as traf
import sent_trafos as sent


def run(dl_dir: str, processed_dir: str):
    splits = ["train", "dev", "test"]
    encodings = ["max", "sentence", "mean", "natural"]

    for enc in encodings:
        enc_dir = os.path.join(processed_dir, enc)
        os.makedirs(enc_dir, exist_ok=True)

        for split in splits:
            print(f"{enc}, {split}")
            split_df_name = os.path.join(dl_dir, f"df_{split}.csv")
            if not os.path.isfile(split_df_name):
                continue
            df = pd.read_csv(split_df_name)
            columns = ["sample"] if "sample" in df.columns else ["reviews_preprocessed"]
            if "label" in df.columns:
                columns += ["label"]
            elif "label_id" in df.columns:
                columns += ["label_id"]

            df = df[columns]
            columns = ["sample", "label"] if len(columns) == 2 else ["sample"]
            df.columns = columns

            Z = joblib.load(os.path.join(dl_dir, f"{split}_rule_matches_z.lib"))
            assert df.shape[0] == Z.shape[0]

            idx = ~df[columns].duplicated()
            df = df[idx]
            Z = Z[idx]

            if enc in ["mean", "max", "natural"]:
                encoded = traf.df_to_sentence_rep(
                    df, column=columns[0], step_size=64, pooling_strategy=enc
                )
            elif enc == "sentence":
                encoded = sent.df_to_sentence_rep(
                    df, column=columns[0], step_size=64
                )
            else:
                raise ValueError(f"Variable enc is not defined correctly.")

            assert encoded.shape[0] == Z.shape[0]
            assert df.shape[0] == Z.shape[0]
            joblib.dump(encoded, os.path.join(enc_dir, f"{split}_tensor.pbz2"))
            joblib.dump(Z, os.path.join(enc_dir, f"{split}_rule_matches_z.pbz2"))
            df.to_csv(os.path.join(enc_dir, f"{split}_df.csv"), index=None, sep="\t")
            if "label" in columns:
                joblib.dump(df['label'].values, os.path.join(enc_dir, f"{split}_labels.pbz2"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dl_dir', type=str)
    parser.add_argument('--processed_dir', type=str)
    parser.add_argument('--type', type=str, default="mean")

    args = parser.parse_args()

    run(dl_dir=args.dl_dir, processed_dir=args.processed_dir)
