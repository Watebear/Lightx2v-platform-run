#!/bin/bash
set -e

# Set these paths for your checkout and model weights.
lightx2v_path=/data/LightX2V
platform_run_path=/data/Lightx2v-platform-run
model_path=/data/models/FLUX.2-dev

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES="${MLU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NEUWARE_HOME="${NEUWARE_HOME:-/usr/local/neuware}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTORCH_MLU_ALLOC_CONF=expandable_segments:True
export DTYPE=BF16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

prompt="Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging \
from the can, captured with sharp detail and natural colors, on a sunlit beach with soft \
shadows and a shallow depth of field, with blurred ocean waves in the background. The can has \
the text 'BFL Diffusers' on it and it has a color gradient that start with #FF5733 at the top \
and transitions to #33FF57 at the bottom."

torchrun --standalone --nproc_per_node=8 -m lightx2v.infer \
    --model_cls flux2 \
    --model-variant dev \
    --task t2i \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/mlu590_scripts/flux2_dev/mlu590/flux2_dev_t2i_1344x768_tp8_8card/config.json" \
    --prompt "${prompt}" \
    --negative_prompt "" \
    --seed 42 \
    --aspect_ratio 16:9 \
    --size 768 1344 \
    --save_result_path "${lightx2v_path}/save_results/flux2_dev_t2i_1344x768_tp8_8card.png" \
    "$@"
