#!/usr/bin/env bash

# environment setup
PROJ_NAME=gmm_nf
PROJ_DIR=/Users/andst/devel/$PROJ_NAME

# variables
EXPERIMENT="mixed"
DATA_DIR=$PROJ_DIR/data
SAVE_DIR="${PROJ_DIR}/data/results/test_${EXPERIMENT}"

# run code
PYTHONPATH=$PROJ_DIR python $PROJ_DIR/scripts/$EXPERIMENT/experiment.py \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --dataset trec \
    --emb_type sentence \
    --option multi_rule \
    --min_matches 200 \
    --depth 2 \
    --label_dim 2 \
    --lr 1e-5 \
    --batch_size 256 \
    --imbalanced_sampling True \
    --num_epochs 4 \
    --num_iters 1
