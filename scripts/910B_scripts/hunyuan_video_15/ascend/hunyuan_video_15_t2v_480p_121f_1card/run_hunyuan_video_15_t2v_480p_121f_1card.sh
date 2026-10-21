#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/HunyuanVideo-1.5
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_HOME_PATH="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"
export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"

prompt="A close-up shot captures a scene on a polished, light-colored granite kitchen counter, \
illuminated by soft natural light from an unseen window. Initially, the frame focuses on a tall, \
clear glass filled with golden, translucent apple juice standing next to a single, shiny red \
apple with a green leaf still attached to its stem. The camera moves horizontally to the right. \
As the shot progresses, a white ceramic plate smoothly enters the frame, revealing a fresh \
arrangement of about seven or eight more apples, a mix of vibrant reds and greens, piled neatly \
upon it. A shallow depth of field keeps the focus sharply on the fruit and glass, while the \
kitchen backsplash in the background remains softly blurred. The scene is in a realistic style."
negative_prompt=""

python -m lightx2v.infer \
    --model_cls hunyuan_video_1.5 \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/910B_scripts/hunyuan_video_15/hunyuan_video_15_t2v_480p_121f_1card/config.json" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --seed 123 \
    --save_result_path "${lightx2v_path}/save_results/hunyuan_video_15_t2v_480p_121f_1card.mp4" \
    "$@"
