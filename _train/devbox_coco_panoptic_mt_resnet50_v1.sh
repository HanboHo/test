#!/usr/bin/env bash

# run from ../models/research

# For logging purposes
CURRENT_TIME=$(date +%Y%m%d%H%M%S)

# Exit immediately if a command exits with a non-zero status.
set -e

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

export CUDA_VISIBLE_DEVICES=3

# Set up the working environment.
# CURRENT_DIR=$(pwd)
OD_DIR="./object_detection"
BASE_DIR="/raid/chen/models"
DATASET="coco"
EVAL_BASE_DIR="/raid/chen/data/01_Coco"

python3 "${OD_DIR}/model_main.py" \
    --logtostderr \
    --pipeline_config_path="${OD_DIR}/_pipeline/devbox_coco_panoptic_mt_resnet50_v1.config" \
    --model_dir="${BASE_DIR}/${DATASET}/devbox_panoptic_mt_resnet50_v1/train_640_estimator_batch4_gpu1" \
    --dataset_name=${DATASET} \
    --num_train_steps=1500000 \
    --num_eval_steps=500 \
    --images_json_file="${EVAL_BASE_DIR}/panoptic_val2017.json" \
    --panoptic_gt_folder="${EVAL_BASE_DIR}/panoptic_val2017" \
    --distribute=False \
    2>&1 | tee -a testing.txt

