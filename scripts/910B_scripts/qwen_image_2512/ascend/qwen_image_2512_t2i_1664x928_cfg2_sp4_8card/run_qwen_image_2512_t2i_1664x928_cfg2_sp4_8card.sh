#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/Qwen-Image-2512
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29500}"
export HCCL_CONNECT_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_HOME_PATH="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"
export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"

prompt="A coffee shop entrance features a chalkboard sign reading \"Qwen Coffee 😊 \$2 per cup,\" with a \
neon light beside it displaying \"通义千问\". Next to it hangs a poster showing a beautiful Chinese \
woman, and beneath the poster is written \"π≈3.1415926-53589793-23846264-33832795-02384197\". \
Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition."
negative_prompt=" "

torchrun --nnodes=1 --nproc_per_node=8 --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" -m lightx2v.infer \
    --model_cls qwen_image \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/910B_scripts/qwen_image_2512/qwen_image_2512_t2i_1664x928_cfg2_sp4_8card/config.json" \
    --prompt "${prompt}" \
    --negative_prompt "${negative_prompt}" \
    --size 928 1664 \
    --aspect_ratio 16:9 \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_2512_t2i_1664x928_cfg2_sp4_8card.png" \
    "$@"
