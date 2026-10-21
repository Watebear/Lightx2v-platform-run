#!/bin/bash
set -e

# System management interface: mx-smi
lightx2v_path=/data/LightX2V
model_path=/data/models/FLUX.2-dev
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"

prompt="Realistic macro photograph of a hermit crab using a soda can as its shell, partially \
emerging from the can, captured with sharp detail and natural colors, on a sunlit \
beach with soft shadows and a shallow depth of field, with blurred ocean waves in \
the background. The can has the text 'BFL Diffusers' on it and it has a color \
gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
--model_cls flux2 \
--model-variant dev \
--task t2i \
--model_path "${model_path}" \
--config_json "${platform_run_path}/scripts/c500_scripts/flux2_dev/flux2_dev_t2i_1344x768_tp8_8card/config.json" \
--prompt "${prompt}" \
--aspect_ratio 16:9 \
--size 768 1344 \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/flux2_dev_t2i_1344x768_tp8_8card.png" \
"$@"
