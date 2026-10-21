#!/bin/bash
set -e

# Set these paths for your checkout and model weights.
lightx2v_path=/data/LightX2V
platform_run_path=/data/Lightx2v-platform-run
model_path=/data/models/Z-Image-Turbo

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0,1}"
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

prompt="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral \
forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds \
round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, \
above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda \
(西安大雁塔), blurred colorful distant lights."

torchrun --standalone --nproc_per_node=2 -m lightx2v.infer \
    --model_cls z_image \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/mlu590_scripts/z_image_turbo/mlu590/z_image_turbo_t2i_1664x928_sp2_2card/config.json" \
    --prompt "${prompt}" \
    --negative_prompt " " \
    --seed 42 \
    --aspect_ratio 16:9 \
    --size 928 1664 \
    --save_result_path "${lightx2v_path}/save_results/z_image_turbo_t2i_1664x928_sp2_2card.png" \
    "$@"
