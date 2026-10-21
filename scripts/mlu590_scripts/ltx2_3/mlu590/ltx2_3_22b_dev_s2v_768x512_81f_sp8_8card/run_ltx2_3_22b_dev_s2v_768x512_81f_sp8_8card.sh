#!/bin/bash
set -e

# Set these paths for your checkout and model weights.
lightx2v_path=/data/LightX2V
platform_run_path=/data/Lightx2v-platform-run
model_path=/data/models/LTX-2

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=${MLU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NEUWARE_HOME=${NEUWARE_HOME:-/usr/local/neuware}
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

prompt="A person speaks clearly in a quiet room, natural lighting, cinematic medium shot."
negative_prompt="blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor \
lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed \
facial features, disfigured hands, artifacts, inconsistent perspective, mismatched lip \
sync, silent or muted audio, distorted voice, robotic voice, echo, off-sync audio, AI \
artifacts."

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --model_cls ltx2 \
    --task ltx2_s2v \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/mlu590_scripts/ltx2_3/mlu590/ltx2_3_22b_dev_s2v_768x512_81f_sp8_8card/config.json" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --seed 42 \
    --audio_path "${lightx2v_path}/assets/inputs/audio/seko_input.mp3" \
    --save_result_path "${lightx2v_path}/save_results/ltx2_3_22b_dev_s2v_768x512_81f_sp8_8card.mp4" \
    "$@"
