#!/bin/bash
set -e

# System management interface: mx-smi
lightx2v_path=/data/LightX2V
model_path=/data/models/LongCat-Image
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"

prompt="一只小猫躺在沙发上"

python -m lightx2v.infer \
--model_cls longcat_image \
--task t2i \
--model_path "${model_path}" \
--config_json "${platform_run_path}/scripts/c500_scripts/longcat_image/longcat_image_t2i_1344x768_1card/config.json" \
--prompt "${prompt}" \
--negative_prompt '' \
--aspect_ratio 16:9 \
--size 768 1344 \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/longcat_image_t2i_1344x768_1card.png" \
"$@"
