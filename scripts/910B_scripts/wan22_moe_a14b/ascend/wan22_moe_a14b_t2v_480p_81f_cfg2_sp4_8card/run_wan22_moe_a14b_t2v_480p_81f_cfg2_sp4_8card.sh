#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/Wan2.2-T2V-A14B
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_HOME_PATH="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"
export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"

prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted \
stage."
negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容\
的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

torchrun --standalone --nnodes=1 --nproc_per_node=8 -m lightx2v.infer \
    --model_cls wan2.2_moe \
    --task t2v \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/910B_scripts/wan22_moe_a14b/wan22_moe_a14b_t2v_480p_81f_cfg2_sp4_8card/config.json" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/wan22_moe_a14b_t2v_480p_81f_cfg2_sp4_8card.mp4" \
    "$@"
