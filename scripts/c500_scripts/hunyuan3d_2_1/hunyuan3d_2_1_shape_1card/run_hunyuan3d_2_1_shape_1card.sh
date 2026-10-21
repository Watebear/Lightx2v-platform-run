#!/bin/bash
set -e

# System management interface: mx-smi
lightx2v_path=/data/LightX2V
model_path=/data/models/Hunyuan3D-2.1
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
--model_cls hunyuan3d \
--task i23d \
--model_path "${model_path}" \
--config_json "${platform_run_path}/scripts/c500_scripts/hunyuan3d_2_1/hunyuan3d_2_1_shape_1card/config.json" \
--image_path /data/Hunyuan3D-2.1-platform/assets/demo.png \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/hunyuan3d_2_1_shape_1card.glb" \
"$@"
