#!/bin/bash
set -e

lightx2v_path=/data/LightX2V
model_path=/data/models/Hunyuan3D-2.1

export PLATFORM=cambricon_mlu
export MLU_VISIBLE_DEVICES=${MLU_VISIBLE_DEVICES:-0}
export NEUWARE_HOME=${NEUWARE_HOME:-/usr/local/neuware}
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${NEUWARE_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export AI_DEVICE=mlu
export DTYPE=FP16
export SENSITIVE_LAYER_DTYPE=None

source "${lightx2v_path}/scripts/base/base.sh"

python "${lightx2v_path}/tools/postprocess/postprocess_paint.py" \
    --hy_repo /data/Hunyuan3D-2.1 \
    --model_path "${model_path}" \
    --mesh_path "${lightx2v_path}/save_results/hunyuan3d/demo.glb" \
    --image_path /data/Hunyuan3D-2.1/assets/demo.png \
    --save_path "${lightx2v_path}/save_results/hunyuan3d/demo_textured.glb" \
    --dino_ckpt_path /data/models/dinov2-giant \
    --realesrgan_ckpt_path /data/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth \
    --render_size 2048 \
    --texture_size 4096 \
    --max_num_view 6 \
    --resolution 512 \
    "$@"
