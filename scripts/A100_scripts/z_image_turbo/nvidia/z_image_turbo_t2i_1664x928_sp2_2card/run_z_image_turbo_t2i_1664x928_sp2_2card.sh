#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/Z-Image-Turbo
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=cuda DEVICE_TYPE=cuda
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export INFER_PROFILE_LEVEL=2

source "${lightx2v_path}/scripts/base/base.sh"

prompt="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead \
pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding \
fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended \
left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred \
colorful distant lights."
negative_prompt=" "

torchrun --nproc_per_node=2 -m lightx2v.infer \
  --model_cls z_image \
  --task t2i \
  --model_path "${model_path}" \
  --config_json "${platform_run_path}/scripts/A100_scripts/z_image_turbo/nvidia/z_image_turbo_t2i_1664x928_sp2_2card/config.json" \
  --seed 42 \
  --prompt "${prompt}" \
  --negative_prompt "${negative_prompt}" \
  --aspect_ratio 16:9 \
  --target_shape 928 1664 \
  --save_result_path "${lightx2v_path}/save_results/z_image_turbo_t2i_1664x928_sp2_2card.png" \
  "$@"
