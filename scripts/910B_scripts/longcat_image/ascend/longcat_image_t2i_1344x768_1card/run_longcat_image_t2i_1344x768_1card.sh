#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/LongCat-Image
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_HOME_PATH="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"
export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"

prompt="一只小猫躺在沙发上"
negative_prompt=""

python -m lightx2v.infer \
    --model_cls longcat_image \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/910B_scripts/longcat_image/longcat_image_t2i_1344x768_1card/config.json" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --size 768 1344 \
    --aspect_ratio 16:9 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/longcat_image_t2i_1344x768_1card.png" \
    "$@"
