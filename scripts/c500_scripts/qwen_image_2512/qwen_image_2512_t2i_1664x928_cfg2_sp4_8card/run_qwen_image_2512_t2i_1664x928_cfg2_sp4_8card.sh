#!/bin/bash
set -e

# System management interface: mx-smi
lightx2v_path=/data/LightX2V
model_path=/data/models/Qwen-Image-2512
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"

prompt="A coffee shop entrance features a chalkboard sign reading \"Qwen Coffee 😊 \$2 per \
cup,\" with a neon light beside it displaying \"通义千问\". Next to it hangs a poster \
showing a beautiful Chinese woman, and beneath the poster is written \
\"π≈3.1415926-53589793-23846264-33832795-02384197\". Ultra HD, 4K, cinematic \
composition, Ultra HD, 4K, cinematic composition."
negative_prompt=" "

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
--model_cls qwen_image \
--task t2i \
--model_path "${model_path}" \
--config_json "${platform_run_path}/scripts/c500_scripts/qwen_image_2512/qwen_image_2512_t2i_1664x928_cfg2_sp4_8card/config.json" \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--aspect_ratio 16:9 \
--size 928 1664 \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/qwen_image_2512_t2i_1664x928_cfg2_sp4_8card.png" \
"$@"
