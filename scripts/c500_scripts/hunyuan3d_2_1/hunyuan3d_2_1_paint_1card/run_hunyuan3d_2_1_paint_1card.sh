#!/bin/bash
set -e

# System management interface: mx-smi
lightx2v_path=/data/LightX2V
model_path=/data/models/Hunyuan3D-2.1
mesh_path="${MESH_PATH:-${lightx2v_path}/save_results/hunyuan3d_2_1_shape_1card.glb}"
image_path=/data/Hunyuan3D-2.1-platform/assets/demo.png

export PLATFORM=metax_cuda
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MACA_PATH=${MACA_PATH:-/opt/maca-3.7.1}
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/ompi/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

source "${lightx2v_path}/scripts/base/base.sh"

# Paint uses the repository's post-processing entry; lightx2v.infer registers Shape only.
# config.json records the matching Paint settings for reference.
python "${lightx2v_path}/tools/postprocess/postprocess_paint.py" \
--hy_repo /data/Hunyuan3D-2.1-platform \
--model_path "${model_path}" \
--mesh_path "${mesh_path}" \
--image_path "${image_path}" \
--dino_ckpt_path /data/models/dinov2-giant \
--realesrgan_ckpt_path "${model_path}/RealESRGAN_x4plus.pth" \
--max_num_view 6 \
--resolution 512 \
--render_size 2048 \
--texture_size 4096 \
--save_path "${lightx2v_path}/save_results/hunyuan3d_2_1_paint_1card.glb" \
"$@"
