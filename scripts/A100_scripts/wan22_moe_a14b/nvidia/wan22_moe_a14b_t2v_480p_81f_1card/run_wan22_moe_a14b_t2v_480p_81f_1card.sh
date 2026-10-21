#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/Wan2.2-T2V-A14B
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=cuda DEVICE_TYPE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export INFER_PROFILE_LEVEL=2

source "${lightx2v_path}/scripts/base/base.sh"

prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a \
spotlighted stage."
negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，\
毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

python -m lightx2v.infer \
  --model_cls wan2.2_moe \
  --task t2v \
  --model_path "${model_path}" \
  --config_json "${platform_run_path}/scripts/A100_scripts/wan22_moe_a14b/nvidia/wan22_moe_a14b_t2v_480p_81f_1card/config.json" \
  --seed 42 \
  --prompt "${prompt}" \
  --negative_prompt "${negative_prompt}" \
  --save_result_path "${lightx2v_path}/save_results/wan22_moe_a14b_t2v_480p_81f_1card.mp4" \
  "$@"
