#!/bin/bash
set -e

# System management interface: mx-smi
lightx2v_path=/data/LightX2V
model_path=/data/models/LTX-2
platform_run_path=/data/Lightx2v-platform-run

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"

prompt="A person speaks clearly in a quiet room, natural lighting, cinematic medium shot."
negative_prompt="blurry, out of focus, overexposed, underexposed, low contrast, excessive noise, poor \
lighting, flickering, motion blur, distorted proportions, unnatural skin tones, \
deformed facial features, disfigured hands, artifacts, inconsistent perspective, \
mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, \
off-sync audio, AI artifacts."

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
--model_cls ltx2 \
--task ltx2_s2v \
--model_path "${model_path}" \
--config_json "${platform_run_path}/scripts/c500_scripts/ltx2_3_22b/ltx2_3_22b_dev_s2v_768x512_81f_sp8_8card/config.json" \
--prompt "${prompt}" \
--negative_prompt "${negative_prompt}" \
--audio_path "${lightx2v_path}/assets/inputs/audio/seko_input.mp3" \
--size 512 768 \
--num_frames 81 \
--seed 42 \
--save_result_path "${lightx2v_path}/save_results/ltx2_3_22b_dev_s2v_768x512_81f_sp8_8card.mp4" \
"$@"
