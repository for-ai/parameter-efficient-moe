#!/usr/bin/env bash

CKPT_DIR=${1:-${CKPT_DIR}}
EVAL_DIR=${2:-${EVAL_DIR}}

T5X_DIR="`python3 -m scripts.find_module t5x`/.."
FLAXFORMER_DIR="`python3 -m scripts.find_module flaxformer`/.."
echo "Searching for gin configs in:"
echo "- ${T5X_DIR}"
echo "- ${FLAXFORMER_DIR}"
echo "============================="
CACHE_DIR="raw_tfrecords/you_cache_dir"

python3 -m t5x.eval \
  --gin_search_paths="${T5X_DIR}" \
  --gin_file="configs/t5/models/t5_1_1_large.gin" \
  --gin_file="configs/ia3_eval.gin" \
  --gin.EVAL_OUTPUT_DIR="'${EVAL_DIR}'" \
  --gin.MIXTURE_OR_TASK_NAME="'t0_eval_score_eval'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 1024, 'targets': 256}" \
  --gin.CHECKPOINT_PATH="'${CKPT_DIR}'" \
  --seqio_additional_cache_dirs=${CACHE_DIR} \
  --gin.utils.DatasetConfig.use_cached="True" \
  --gin.utils.DatasetConfig.split="'validation'" \
  --gin.BATCH_SIZE="32" 