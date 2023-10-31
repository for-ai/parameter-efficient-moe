#!/usr/bin/env bash

MODEL_DIR=${1:-${MODEL_DIR}}

T5X_DIR="`python3 -m scripts.find_module t5x`/.."
FLAXFORMER_DIR="`python3 -m scripts.find_module flaxformer`/.."
echo "Searching for gin configs in:"
echo "- ${T5X_DIR}"
echo "- ${FLAXFORMER_DIR}"
echo "============================="
PRETRAINED_MODEL="gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_large/checkpoint_1100000"
CACHE_DIR="raw_tfrecords/you_cache_dir"

python3 -m t5x.train \
  --gin_search_paths="${T5X_DIR}" \
  --gin_file="configs/t5/models/t5_1_1_large.gin" \
  --gin_file="configs/ia3.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.LOSS_NORMALIZING_FACTOR="'AVERAGE_PER_SEQUENCE'" \
  --gin.MIXTURE_OR_TASK_NAME="'t0_train'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 1024, 'targets': 256}" \
  --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
  --gin.TRAIN_STEPS="1_600_000" \
  --gin.USE_CACHED_TASKS="True" \
  --gin.PACKING="True" \
  --seqio_additional_cache_dirs=${CACHE_DIR} \
  --gin.BATCH_SIZE="32" 