#!/bin/bash
set -e

# System management interface: mx-smi
lightx2v_path=/data/LightX2V
model_path=/data/models/Z-Image-Turbo
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"

prompt="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red \
floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, \
beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp \
(⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night \
background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
negative_prompt=" "

python -m lightx2v.infer \
--model_cls z_image \
--task t2i \
--model_path "${model_path}" \
--config_json "${platform_run_path}/scripts/c500_scripts/z_image_turbo/z_image_turbo_t2i_1664x928_1card/config.json" \
--prompt "${prompt}" \
--aspect_ratio 16:9 \
--size 928 1664 \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/z_image_turbo_t2i_1664x928_1card.png" \
"$@"
