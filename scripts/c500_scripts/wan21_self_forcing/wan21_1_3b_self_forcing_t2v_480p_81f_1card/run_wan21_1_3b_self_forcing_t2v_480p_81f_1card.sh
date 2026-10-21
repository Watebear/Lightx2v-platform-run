#!/bin/bash
set -e

# System management interface: mx-smi
lightx2v_path=/data/LightX2V
model_path=/data/models/Wan2.1-T2V-1.3B
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"

prompt="A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights \
and animated city signs casting vibrant reflections. She wears a sleek black leather \
jacket paired with a flowing red dress and black boots, her black purse slung over \
her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her \
confident, casual demeanor. The street is damp and reflective, creating a \
mirror-like effect that enhances the colorful lights and shadows. Pedestrians move \
about, adding to the lively atmosphere. The scene is captured in a dynamic medium \
shot with the woman walking slightly to one side, highlighting her graceful strides."

python -m lightx2v.infer \
--model_cls wan2.1_sf \
--task t2v \
--model_path "${model_path}" \
--config_json "${platform_run_path}/scripts/c500_scripts/wan21_self_forcing/wan21_1_3b_self_forcing_t2v_480p_81f_1card/config.json" \
--prompt "${prompt}" \
--size 480 832 \
--num_frames 81 \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/wan21_1_3b_self_forcing_t2v_480p_81f_1card.mp4" \
"$@"
