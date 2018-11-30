#!/usr/bin/env bash

# run from ../models/research

# For logging purposes
CURRENT_TIME=$(date +%Y%m%d%H%M%S)

# Exit immediately if a command exits with a non-zero status.
set -e

# Update PYTHONPATH.
export PYTHONPATH=${PYTHONPATH}:`pwd`:`pwd`/slim

export CUDA_VISIBLE_DEVICES=1

# Set up the working environment.
OD_DIR="./object_detection"
DATASET="cityscapes"
MODEL="docker_dgx3_panoptic_mt_resnet50_v1"
MODEL_DATA="docker_dgx3_cityscapes_panoptic_mt_resnet50_v1"
SPEC="train_${2}_estimator_batch${1}_gpu1"

BASE_SAVE_PATH="/remote/02_Cityscapes"

MODEL_SAVE_PATH="${BASE_SAVE_PATH}/model/${MODEL}/${SPEC}_${CURRENT_TIME}"
MODEL_RESTORE_PATH=""
CONFIG_PATH="${OD_DIR}/_pipeline/${MODEL_DATA}.config"

# MODEL_SAVE_PATH="${BASE_SAVE_PATH}/model/${MODEL}/train_512256_estimator_batch16_gpu1_20181101135554_part4"
# MODEL_RESTORE_PATH=""
# CONFIG_PATH="${MODEL_SAVE_PATH}/pipeline.config"

# Save training script
mkdir ${MODEL_SAVE_PATH}
cp "${OD_DIR}/_train/${MODEL_DATA}.sh" ${MODEL_SAVE_PATH}
cp "${OD_DIR}/meta_architectures/mt_meta_arch.py" ${MODEL_SAVE_PATH}

chen-python3 "${OD_DIR}/model_main.py" \
    --logtostderr \
    --pipeline_config_path=${CONFIG_PATH} \
    --model_dir=${MODEL_SAVE_PATH} \
    --restore_dir=${MODEL_RESTORE_PATH} \
    --dataset_name=${DATASET} \
    --num_train_steps=90000 \
    --num_eval_steps=500 \
    --log_step_count_steps=10 \
    --save_checkpoints_secs=3600 \
    --save_summary_steps=500 \
    --throttle_secs=1800 \
    --start_delay_secs=3600 \
    --images_json_file="${BASE_SAVE_PATH}/data/panoptic_val2017.json" \
    --panoptic_gt_folder="${BASE_SAVE_PATH}/data/panoptic_val2017" \
    --distribute=False
