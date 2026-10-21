#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/Hunyuan3D-2.1
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=cuda DEVICE_TYPE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INFER_PROFILE_LEVEL=2

source "${lightx2v_path}/scripts/base/base.sh"

# base.sh sets BF16; Hunyuan3D uses FP16.
export DTYPE=FP16

python -m lightx2v.infer \
  --model_cls hunyuan3d \
  --task i23d \
  --model_path "${model_path}" \
  --config_json "${platform_run_path}/scripts/A100_scripts/hunyuan3d_2_1/nvidia/hunyuan3d-2.1-shape_1card/config.json" \
  --seed 42 \
  --image_path /data/inputs/hunyuan3d/demo.png \
  --save_result_path "${lightx2v_path}/save_results/hunyuan3d-2.1-shape_1card.glb" \
  "$@"
