#!/usr/bin/env bash

# run from ../models/research

# For logging purposes
CURRENT_TIME=$(date +%Y%m%d%H%M%S)

# Exit immediately if a command exits with a non-zero status.
set -e

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

export CUDA_VISIBLE_DEVICES=4

# Set up the working environment.
# CURRENT_DIR=$(pwd)
OD_DIR="./object_detection"
BASE_DIR="/panoptic/models"

chen-python3 "${OD_DIR}/model_main.py" \
    --logtostderr \
    --pipeline_config_path="${OD_DIR}/_pipeline/docker_dgx3_coco_panoptic_mask_rcnn_fpn_resnet101_v1.config" \
    --model_dir="${BASE_DIR}/docker_panoptic_mask_rcnn_fpn_resnet101_v1/train_dgx3_768_estimator_batch3_gpu1_mixprecision" \
    --dataset_name="coco" \
    --num_train_steps=1500000 \
    --num_eval_steps=500 \
    --images_json_file="/panoptic/data/mscoco/validation/panoptic/panoptic_val2017.json" \
    --panoptic_gt_folder="/panoptic/data/mscoco/validation/panoptic" \
    --batch_size=3 \
    --distribute=false \
    2>&1 | tee -a testing.txt

