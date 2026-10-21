#!/bin/bash
set -e

# Set these paths for your checkout and model weights.
lightx2v_path=/data/LightX2V
platform_run_path=/data/Lightx2v-platform-run
model_path=/data/models/Qwen-Image-2512

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

prompt="A coffee shop entrance features a chalkboard sign reading \"Qwen Coffee 😊 \$2 per cup,\" with a \
neon light beside it displaying \"通义千问\". Next to it hangs a poster showing a beautiful Chinese \
woman, and beneath the poster is written \"π≈3.1415926-53589793-23846264-33832795-02384197\". \
Ultra HD, 4K, cinematic composition, Ultra HD, 4K, cinematic composition."

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --model_cls qwen_image \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/mlu590_scripts/qwen_image/mlu590/qwen_image_2512_t2i_1664x928_cfg2_sp4_8card/config.json" \
    --prompt "${prompt}" \
    --negative_prompt " " \
    --seed 42 \
    --aspect_ratio 16:9 \
    --size 928 1664 \
    --save_result_path "${lightx2v_path}/save_results/qwen_image_2512_t2i_1664x928_cfg2_sp4_8card.png" \
    "$@"
