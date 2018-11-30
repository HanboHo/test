#!/usr/bin/env bash

# run from ../models/research

# For logging purposes
CURRENT_TIME=$(date +%Y%m%d%H%M%S)

# Exit immediately if a command exits with a non-zero status.
set -e

# Update PYTHONPATH.
export PYTHONPATH=${PYTHONPATH}:`pwd`:`pwd`/slim

export CUDA_VISIBLE_DEVICES=7

# Set up the working environment.
OD_DIR="./object_detection"
DATASET="coco"
MODEL="docker_dgx3_panoptic_mt_resnet50_v1"
MODEL_DATA="docker_dgx3_coco_panoptic_mt_resnet50_v1"
SPEC="train_640_estimator_batch$1_gpu1"

BASE_SAVE_PATH="/remote/01_Coco"
MODEL_SAVE_PATH="${BASE_SAVE_PATH}/model/${MODEL}/${SPEC}_${CURRENT_TIME}"
CONFIG_PATH="${OD_DIR}/_pipeline/${MODEL_DATA}.config"

# Save training script
mkdir ${MODEL_SAVE_PATH}
cp "${OD_DIR}/_train/${MODEL_DATA}.sh" ${MODEL_SAVE_PATH}

chen-python3 "${OD_DIR}/model_main.py" \
    --logtostderr \
    --pipeline_config_path=${CONFIG_PATH} \
    --model_dir=${MODEL_SAVE_PATH} \
    --dataset_name=${DATASET} \
    --num_train_steps=1500000 \
    --num_eval_steps=500 \
    --images_json_file="${BASE_SAVE_PATH}/data/panoptic_val2017.json" \
    --panoptic_gt_folder="${BASE_SAVE_PATH}/data/panoptic_val2017" \
    --distribute=False
