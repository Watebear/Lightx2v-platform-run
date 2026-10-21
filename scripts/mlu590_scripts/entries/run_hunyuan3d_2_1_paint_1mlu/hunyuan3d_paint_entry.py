#!/usr/bin/env python3
"""MLU590 Hunyuan3D Paint two-request entry.

Paint is a separate pipeline in this LightX2V revision.  This adapter keeps
the same renderer, multiview diffusion, super-resolution and texture bake in
both modes.  The no-save mode skips only final OBJ/GLB export.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
from types import MethodType, SimpleNamespace


def no_export_call(original):
    """Rebuild the installed upstream call, removing exactly its two exports."""
    method = inspect.unwrap(original)
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    node = tree.body[0]
    node.decorator_list = []
    removed = 0
    body = []
    for statement in node.body:
        is_obj_export = (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "save_mesh"
        )
        is_glb_export = (
            isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Name)
            and statement.test.id == "save_glb"
        )
        if is_obj_export or is_glb_export:
            removed += 1
        else:
            body.append(statement)
    if removed != 2:
        raise RuntimeError("Hunyuan3D Paint export source changed; review adapter")
    node.body = body
    ast.fix_missing_locations(tree)
    scope = dict(method.__globals__)
    exec(compile(tree, inspect.getsourcefile(method), "exec"), scope)
    return scope[node.name]


def distributed_barrier():
    import torch
    import torch.distributed as dist
    from loguru import logger

    torch.mlu.synchronize()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
    logger.info("[Benchmark] between-request distributed_barrier() returned")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mode", required=True, choices=("no_save", "save"))
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    import torch
    import torch_mlu  # noqa: F401
    from loguru import logger
    from torch._dynamo.backends.registry import lookup_backend

    from lightx2v.utils.profiler import ProfilingContext4DebugL1
    from lightx2v.utils.utils import seed_all

    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    config["model_path"] = args.model_path
    required = (
        "model_path", "hy_repo", "mesh_path", "image_path", "dino_ckpt_path",
        "realesrgan_ckpt_path", "render_size", "texture_size", "max_num_view",
        "resolution", "seed", "use_remesh", "use_compile", "warmup",
    )
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Paint config missing: {missing}")
    if not config["use_compile"] or not config["warmup"]:
        raise ValueError("Paint benchmark requires warmup and compile")
    if not torch.mlu.is_available() or torch.mlu.device_count() < 1:
        raise RuntimeError("No MLU is visible to Paint")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raster_root = Path(config["hy_repo"]) / "hy3dpaint" / "custom_rasterizer"
    if not (raster_root / "custom_rasterizer" / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing Paint rasterizer package: {raster_root}")
    sys.path.insert(0, str(raster_root))
    import custom_rasterizer
    if not callable(getattr(custom_rasterizer, "rasterize", None)):
        raise RuntimeError(f"Invalid custom_rasterizer import: {custom_rasterizer.__file__}")
    logger.info("[Benchmark] Paint rasterizer loaded from {}", custom_rasterizer.__file__)

    postprocess_dir = Path(os.environ["LIGHTX2V_SOURCE"]) / "tools" / "postprocess"
    module_path = postprocess_dir / "paint_pipeline.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Missing PaintPipeline source: {module_path}")
    if str(postprocess_dir) not in sys.path:
        sys.path.insert(0, str(postprocess_dir))
    spec = importlib.util.spec_from_file_location("h3d_paint_pipeline", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PaintPipeline source: {module_path}")
    paint_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paint_module)
    PaintPipeline = paint_module.PaintPipeline
    _profile_paint_models = paint_module._profile_paint_models
    logger.info("[Benchmark] PaintPipeline loaded from {}", module_path)

    pipeline = PaintPipeline(
        **{key: config[key] for key in (
            "model_path", "hy_repo", "dino_ckpt_path", "realesrgan_ckpt_path",
            "render_size", "texture_size", "max_num_view", "resolution",
        )},
        device="mlu:0",
    )
    native = pipeline._pipeline
    diffusion = native.models["multiview_model"].pipeline
    original_unet_forward = diffusion.unet.forward
    original_native_call = type(native).__call__
    no_save_call = torch.no_grad()(no_export_call(original_native_call))
    state = {"graphs": 0, "graph_executions": 0, "attempts": 0, "completed": 0,
             "dtypes": set()}
    inductor = lookup_backend("inductor")

    def checked_backend(graph_module, example_inputs):
        tensors = [value for value in example_inputs if isinstance(value, torch.Tensor)]
        if not any(value.device.type == "mlu" for value in tensors):
            raise RuntimeError("Paint compiled graph has no MLU tensor")
        executable = inductor(graph_module, example_inputs)
        state["graphs"] += 1
        logger.info("[H3DCompile] Paint compiled MLU graph {} inputs={}", state["graphs"],
                    [(tuple(value.shape), str(value.dtype), str(value.device)) for value in tensors])

        def execute(*inputs, **kwargs):
            result = executable(*inputs, **kwargs)
            state["graph_executions"] += 1
            return result

        return execute

    compiled_unet = torch.compile(original_unet_forward, backend=checked_backend,
                                  fullgraph=False, dynamic=False)

    def unet_forward(*inputs, **kwargs):
        state["attempts"] += 1
        for value in inputs:
            if isinstance(value, torch.Tensor) and value.device.type == "mlu":
                state["dtypes"].add(str(value.dtype))
        result = compiled_unet(*inputs, **kwargs)
        state["completed"] += 1
        return result

    diffusion.unet.forward = unet_forward
    logger.info("[H3DCompile] Paint multiview UNet forward compile installed")

    def evidence(stage):
        logger.info(
            "[H3DCompile] evidence stage={} scope=paint.multiview.unet.forward "
            "attempts={} completed={} graphs={} graph_executions={} mlu_dtypes={}",
            stage, state["attempts"], state["completed"], state["graphs"],
            state["graph_executions"], sorted(state["dtypes"]),
        )

    with tempfile.TemporaryDirectory(prefix="paint_input_", dir=output_dir) as scratch:
        mesh = Path(scratch) / "input.glb"
        shutil.copy2(config["mesh_path"], mesh)
        image = str(config["image_path"])
        use_remesh = bool(config["use_remesh"])

        def run_without_export():
            # Match PaintPipeline.__call__ in save mode: its model profiler
            # synchronizes each multiview/super-resolution call at level 1.
            with _profile_paint_models(native, "mlu:0"):
                return no_save_call(native, mesh_path=str(mesh), image_path=image,
                                    output_mesh_path=None, use_remesh=use_remesh,
                                    save_glb=False)

        # Paint has no native runner or warmup method.  A complete pipeline
        # pass warms model, renderer and compiler without writing final media.
        seed_all(int(config["seed"]))
        logger.info("[Benchmark] warmup begin seed={} save_result_path=None", config["seed"])
        with ProfilingContext4DebugL1("RUN pipeline"):
            result = run_without_export()
            del result
        logger.info("[Benchmark] warmup end")
        evidence("after_warmup")

        class PaintRunner:
            def run_request(self, input_info):
                request = input_info.request_number
                seed_all(input_info.seed)
                logger.info("[Benchmark] request {} begin rank=0 seed={} save_result_path={}",
                            request, input_info.seed, input_info.save_result_path)
                with ProfilingContext4DebugL1("RUN pipeline"):
                    if input_info.save_result_path is None:
                        result = run_without_export()
                    else:
                        result = pipeline(
                            mesh_path=str(mesh), image_path=image,
                            save_path=input_info.save_result_path,
                            use_remesh=use_remesh, save_glb=True,
                        )
                    del result
                logger.info("[Benchmark] request {} end rank=0", request)
                evidence(f"after_request_{request}")

        runner = PaintRunner()
        for request_number in (1, 2):
            save_path = (
                str(output_dir / f"request_{request_number}" / "textured.glb")
                if args.mode == "save" else None
            )
            if save_path is not None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=False)
            input_info = SimpleNamespace(
                request_number=request_number,
                seed=int(config["seed"]),
                save_result_path=save_path,
            )
            runner.run_request(input_info)
            if request_number == 1:
                distributed_barrier()


if __name__ == "__main__":
    if "--benchmark_mode" in sys.argv:
        from benchmark_run import main as benchmark_main

        raise SystemExit(benchmark_main(sys.argv[1:]))
    main()
