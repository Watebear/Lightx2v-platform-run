#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/LongCat-Image
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=cuda DEVICE_TYPE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INFER_PROFILE_LEVEL=2

source "${lightx2v_path}/scripts/base/base.sh"

prompt="一只小猫躺在沙发上"
negative_prompt=""

python -m lightx2v.infer \
  --model_cls longcat_image \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${platform_run_path}/scripts/A100_scripts/longcat_image/nvidia/longcat_image_t2i_1344x768_1card/config.json" \
  --seed 42 \
  --prompt "${prompt}" \
  --negative_prompt "${negative_prompt}" \
  --aspect_ratio 16:9 \
  --target_shape 768 1344 \
  --save_result_path "${lightx2v_path}/save_results/longcat_image_t2i_1344x768_1card.png" \
  "$@"
