#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/LTX-2
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_HOME_PATH="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"
export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"

prompt="A person speaks clearly in a quiet room, natural lighting, cinematic medium shot."
negative_prompt="blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor lighting, \
flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, \
disfigured hands, artifacts, inconsistent perspective, mismatched lip sync, silent or muted \
audio, distorted voice, robotic voice, echo, off-sync audio, AI artifacts."

python -m lightx2v.infer \
    --model_cls ltx2 \
    --task ltx2_s2v \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/910B_scripts/ltx2_3_22b/ltx2_3_22b_dev_s2v_768x512_81f_1card/config.json" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --audio_path "/data/inputs/audio/seko_input.mp3" \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/ltx2_3_22b_dev_s2v_768x512_81f_1card.mp4" \
    "$@"
