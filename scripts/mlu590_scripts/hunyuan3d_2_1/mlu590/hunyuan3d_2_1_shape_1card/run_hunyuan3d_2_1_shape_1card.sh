#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
platform_run_path=/data/Lightx2v-platform-run
model_path=/data/models/Hunyuan3D-2.1
hunyuan3d_path=/data/Hunyuan3D-2.1

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=${MLU_VISIBLE_DEVICES:-0}
export NEUWARE_HOME=${NEUWARE_HOME:-/usr/local/neuware}
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export AI_DEVICE=mlu
export DTYPE=FP16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

python -m lightx2v.infer \
    --model_cls hunyuan3d \
    --task i23d \
    --model_path "${model_path}" \
    --config_json "${platform_run_path}/scripts/mlu590_scripts/hunyuan3d_2_1/mlu590/hunyuan3d_2_1_shape_1card/config.json" \
    --image_path "${hunyuan3d_path}/assets/demo.png" \
    --seed 42 \
    --save_result_path "${lightx2v_path}/save_results/hunyuan3d/demo.glb" \
    "$@"
