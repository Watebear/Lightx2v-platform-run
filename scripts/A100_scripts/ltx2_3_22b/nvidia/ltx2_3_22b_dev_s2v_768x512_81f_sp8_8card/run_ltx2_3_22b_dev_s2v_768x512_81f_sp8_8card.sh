#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/LTX-2.3
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=cuda DEVICE_TYPE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export INFER_PROFILE_LEVEL=2

source "${lightx2v_path}/scripts/base/base.sh"

prompt="A person speaks clearly in a quiet room, natural lighting, cinematic medium shot."
negative_prompt="blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, \
flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, \
disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted \
audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts."

torchrun --nproc_per_node=8 -m lightx2v.infer \
  --model_cls ltx2 \
  --task ltx2_s2v \
  --model_path "${model_path}" \
  --config_json "${platform_run_path}/scripts/A100_scripts/ltx2_3_22b/nvidia/ltx2_3_22b_dev_s2v_768x512_81f_sp8_8card/config.json" \
  --seed 42 \
  --prompt "${prompt}" \
  --negative_prompt "${negative_prompt}" \
  --audio_path /data/inputs/audio/seko_input.mp3 \
  --save_result_path "${lightx2v_path}/save_results/ltx2_3_22b_dev_s2v_768x512_81f_sp8_8card.mp4" \
  "$@"
