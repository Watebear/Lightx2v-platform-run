#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/Hunyuan3D-2.1

export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_HOME_PATH="${CANN_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_OPP_PATH="${ASCEND_HOME_PATH}/opp"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export AI_DEVICE=npu
export DTYPE=FP16
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HOME}/.cache/huggingface/modules}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}"

source "${lightx2v_path}/scripts/base/base.sh"
export PYTHONPATH="${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe:${PYTHONPATH}"

python "${lightx2v_path}/tools/postprocess/postprocess_paint.py" \
    --hy_repo /data/Hunyuan3D-2.1-platform \
    --model_path "${model_path}" \
    --mesh_path /data/inputs/meshes/hunyuan3d_shape_input.glb \
    --image_path /data/inputs/hunyuan3d/demo.png \
    --save_path "${lightx2v_path}/save_results/hunyuan3d_2_1_paint_1card.glb" \
    --device npu \
    --max_num_view 6 \
    --resolution 512 \
    --render_size 2048 \
    --texture_size 4096 \
    "$@"
