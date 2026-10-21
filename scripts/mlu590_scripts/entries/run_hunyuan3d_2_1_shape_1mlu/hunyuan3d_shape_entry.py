#!/usr/bin/env python3
"""Hunyuan3D Shape benchmark with warmup and two requests."""

from __future__ import annotations

import ast
import argparse
import inspect
import os
from pathlib import Path
import textwrap


STATE = {"attempts": 0, "completed": 0, "graphs": 0, "graph_executions": 0, "dtypes": set()}


def _run_main_without_save_guard(original):
    """Keep the exact source method apart from its mandatory save-path guard."""
    unwrapped = inspect.unwrap(original)
    tree = ast.parse(textwrap.dedent(inspect.getsource(unwrapped)))
    method = tree.body[0]
    method.decorator_list = []
    guards = [
        statement for statement in method.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.UnaryOp)
        and isinstance(statement.test.op, ast.Not)
        and isinstance(statement.test.operand, ast.Name)
        and statement.test.operand.id == "save_result_path"
    ]
    if len(guards) != 1:
        raise RuntimeError("Hunyuan3D Shape source save-path guard changed; review adapter")
    method.body = [statement for statement in method.body if statement is not guards[0]]
    ast.fix_missing_locations(tree)
    scope = dict(unwrapped.__globals__)
    exec(compile(tree, inspect.getsourcefile(unwrapped), "exec"), scope)
    return scope[method.name]


def install_shape_adapters(config):
    if config["model_cls"] != "hunyuan3d":
        raise ValueError("This entry only supports Hunyuan3D Shape")
    if not config.get("use_compile"):
        raise ValueError("Hunyuan3D Shape compile must be requested")

    import torch
    from loguru import logger
    from torch._dynamo.backends.registry import lookup_backend
    from lightx2v.common.ops.moe.fused_moe import _TorchFusedMoE
    from lightx2v.models.networks.hunyuan3d.infer.pre_infer import Hunyuan3DPreInfer
    from lightx2v.models.runners.hunyuan3d.hunyuan3d_shape_runner import Hunyuan3DShapeRunner
    from lightx2v.utils.profiler import ProfilingContext4DebugL2

    # This source revision converts the MLU route index to int32 for
    # index_select, then reuses it in index_copy_, which requires int64 in
    # the installed torch_mlu. Keep the original routing and convert only the
    # returned index used by index_copy_.
    original_permute_routes = _TorchFusedMoE._permute_routes

    def permute_routes_local(self, input, token_selected_experts):
        permuted_input, permuted_to_expanded, permuted_experts, counts = (
            original_permute_routes(self, input, token_selected_experts)
        )
        if input.device.type == "mlu" and permuted_to_expanded.dtype != torch.long:
            permuted_to_expanded = permuted_to_expanded.to(torch.long)
        return permuted_input, permuted_to_expanded, permuted_experts, counts

    _TorchFusedMoE._permute_routes = permute_routes_local
    logger.info("[H3DShape] MoE route index adapter installed for MLU index_copy_")

    # This source revision does not call init_compile() for Hunyuan3D at all;
    # its use_compile setting alone has no effect.  Compile the real
    # timestep/latent embedding stage, which runs on each DiT pass.
    logger.info("[H3DCompile] native Hunyuan3D compile path absent; pre_infer.infer selected")

    inductor = lookup_backend("inductor")

    def checked_backend(graph_module, example_inputs):
        tensors = [value for value in example_inputs if isinstance(value, torch.Tensor)]
        if not any(value.device.type == "mlu" for value in tensors):
            raise RuntimeError("Hunyuan3D Shape compiled graph has no MLU tensor")
        executable = inductor(graph_module, example_inputs)
        STATE["graphs"] += 1
        logger.info("[H3DCompile] compiled MLU graph {} inputs={}", STATE["graphs"],
                    [(tuple(value.shape), str(value.dtype), str(value.device)) for value in tensors])

        def execute(*args, **kwargs):
            result = executable(*args, **kwargs)
            STATE["graph_executions"] += 1
            return result

        return execute

    original_pre_infer = Hunyuan3DPreInfer.infer
    compiled_pre_infer = torch.compile(original_pre_infer, backend=checked_backend,
                                       fullgraph=False, dynamic=False)

    def infer_local(self, weights, hidden_states, cond, timestep, guidance_cond=None):
        if hidden_states.device.type != "mlu":
            raise RuntimeError("Hunyuan3D Shape pre-infer expected an MLU tensor")
        STATE["attempts"] += 1
        STATE["dtypes"].add(str(hidden_states.dtype))
        result = compiled_pre_infer(self, weights, hidden_states, cond, timestep, guidance_cond)
        STATE["completed"] += 1
        return result

    Hunyuan3DPreInfer.infer = infer_local

    Hunyuan3DShapeRunner.run_main = ProfilingContext4DebugL2("Run DiT")(
        _run_main_without_save_guard(Hunyuan3DShapeRunner.run_main)
    )
    original_save_mesh = Hunyuan3DShapeRunner._save_mesh

    def save_mesh_local(self, mesh, path):
        if path is None:
            logger.info("[H3DShape] mesh decoded; save_result_path=None; export skipped")
            return mesh
        return original_save_mesh(self, mesh, path)

    Hunyuan3DShapeRunner._save_mesh = save_mesh_local
    logger.info("[H3DShape] no-save adapter installed; full mesh decode retained")


def shape_compile_evidence(stage):
    from loguru import logger

    logger.info(
        "[H3DCompile] evidence stage={} scope=pre_infer.infer attempts={} completed={} "
        "graphs={} graph_executions={} mlu_dtypes={}",
        stage, STATE["attempts"], STATE["completed"], STATE["graphs"],
        STATE["graph_executions"], sorted(STATE["dtypes"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--return_result_tensor", action="store_true")
    parser.add_argument("--save_result_path")
    args = parser.parse_args()

    import torch
    import torch_mlu  # noqa: F401
    from loguru import logger
    from lightx2v.infer import distributed_barrier
    from lightx2v.models.runners.runner_factory import build_runner
    from lightx2v.utils.set_config import build_startup_config
    from lightx2v.utils.utils import validate_config_paths

    config = build_startup_config({
        "model_cls": args.model_cls,
        "task": args.task,
        "model_path": args.model_path,
        "config_json": args.config_json,
    })
    warmup = args.warmup or config.get("warmup", False)
    config["warmup"] = False  # Hunyuan3D runs a full request below.
    validate_config_paths(config)
    install_shape_adapters(config)
    runner = build_runner(config)

    def request(save_path):
        return runner.prepare_request({
            "task": args.task,
            "image_path": args.image_path,
            "seed": args.seed,
            "save_result_path": save_path,
        })

    if warmup:
        logger.info("[Benchmark] warmup begin seed={} save_result_path=None", args.seed)
        result = runner.run_request(request(None))
        del result
        torch.mlu.synchronize()
        logger.info("[Benchmark] warmup end")
    shape_compile_evidence("after_warmup")

    save_target = Path(args.save_result_path) if args.save_result_path else None
    for request_number in (1, 2):
        save_path = (str(save_target.with_name(f"request_{request_number}{save_target.suffix}"))
                     if save_target else None)
        input_info = request(save_path)
        logger.info("[Benchmark] request {} begin rank={} seed={} save_result_path={}",
                    request_number, os.environ.get("RANK", "0"), input_info.seed, save_path)
        result = runner.run_request(input_info)
        del result
        torch.mlu.synchronize()
        logger.info("[Benchmark] request {} end rank={}", request_number, os.environ.get("RANK", "0"))
        shape_compile_evidence(f"after_request_{request_number}")
        if request_number == 1:
            distributed_barrier()
            logger.info("[Benchmark] between-request distributed_barrier() returned")


if __name__ == "__main__":
    main()
