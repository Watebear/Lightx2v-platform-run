#!/bin/bash
set -e

# Set these paths for your checkout and model weights.
lightx2v_path=/data/LightX2V
platform_run_path=/data/Lightx2v-platform-run
model_path=/data/models/LongCat-Image

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0}"
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

prompt="一只小猫躺在沙发上"

python -m lightx2v.infer \
    --model_cls longcat_image \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/mlu590_scripts/longcat_image/mlu590/longcat_image_t2i_1344x768_1card/config.json" \
    --prompt "${prompt}" \
    --negative_prompt "" \
    --seed 42 \
    --aspect_ratio 16:9 \
    --size 768 1344 \
    --save_result_path "${lightx2v_path}/save_results/longcat_image_t2i_1344x768_1card.png" \
    "$@"
