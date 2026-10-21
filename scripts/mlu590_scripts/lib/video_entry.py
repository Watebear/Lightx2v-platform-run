"""Run video benchmarks against the LightX2V inference API."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from video_compile import (
    feature_compile_evidence,
    gelu_graph_counters,
    install_activation_compile,
    install_wan_gelu_only_compile,
    register_ltx_gemma_attention_alias,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--audio_path", default=None)
    parser.add_argument("--size", nargs=2, type=int, default=None)
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--save_result_path", default=None)
    parser.add_argument("--return_result_tensor", action="store_true")
    return parser.parse_args()


def transformer_states(runner, compile_kind):
    states = []
    seen = set()
    candidates = list(runner.__dict__.items())
    for parent_name, parent in list(candidates):
        children = getattr(parent, "model", None)
        if isinstance(children, (list, tuple)):
            candidates.extend((f"{parent_name}.model[{index}]", child)
                              for index, child in enumerate(children))
    for name, model in candidates:
        transformer = getattr(model, "transformer_infer", None)
        if transformer is None or id(transformer) in seen:
            continue
        seen.add(id(transformer))
        blocks = getattr(transformer, "compiled_blocks", None)
        block_count = len(blocks) if isinstance(blocks, dict) else None
        base = (name, type(transformer).__name__, getattr(transformer, "use_compile", None), block_count)
        if compile_kind == "wan_gelu":
            cache = getattr(transformer, "_benchmark_gelu_compiled", {})
            states.append(base + (
                len(cache),
                sum(item["attempts"] for item in cache.values()),
                sum(item["completed"] for item in cache.values()),
                sum(item["inductor_graphs"] for item in cache.values()),
            ))
        else:
            states.append(base)
    return states


def run_main(compile_kind):
    args = parse_args()
    if not args.warmup:
        raise ValueError("These benchmark cases require --warmup")

    from lightx2v import infer
    from lightx2v.utils.profiler import torch_device_module
    from lightx2v.utils.set_config import build_cli_inputs, init_parallel, print_config, print_request
    from lightx2v.utils.utils import validate_config_paths
    from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

    config, request_data = build_cli_inputs(args)
    request_data.pop("warmup", None)
    if request_data.get("negative_prompt") == "":
        request_data.pop("negative_prompt")

    native_warmup = compile_kind in {"wan_gelu", "native_blocks"}
    config["warmup"] = native_warmup
    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER[os.environ["PLATFORM"]]
        platform_device.init_parallel_env()
        init_parallel(config)
    print_config(config, title="Startup config")
    validate_config_paths(config)

    if compile_kind == "wan_gelu":
        install_wan_gelu_only_compile(config)
    elif compile_kind == "feature":
        register_ltx_gemma_attention_alias(config)
        install_activation_compile(config)
    elif compile_kind != "native_blocks":
        raise ValueError(f"Unsupported compile strategy: {compile_kind}")

    try:
        runner = infer.build_runner(config)
        logger.info("[Benchmark] native warmup lifecycle returned enabled={} done={}",
                    native_warmup, runner._warmup_done)

        if not native_warmup:
            warmup_data = request_data.copy()
            warmup_data["save_result_path"] = None
            warmup_data["return_result_tensor"] = True
            warmup_info = runner.prepare_request(warmup_data)
            scheduler = getattr(runner, "scheduler", None)
            if scheduler is not None and hasattr(scheduler, "generator"):
                scheduler.generator = None
            logger.info("[Benchmark] warmup begin seed={} save_result_path=None", warmup_info.seed)
            result = runner.run_request(warmup_info)
            del result
            torch_device_module.synchronize()
            infer.distributed_barrier()
            logger.info("[Benchmark] warmup end")

        def evidence(stage):
            states = transformer_states(runner, compile_kind)
            if compile_kind == "wan_gelu":
                logger.info(
                    "[BenchmarkGELU] compile evidence stage={} config_use_compile={} "
                    "transformer_states=(name,class,use_compile,full_block_cache,gelu_cache,gelu_attempts,gelu_completed,inductor_graphs): {} graph_counters={}",
                    stage, config.get("use_compile", False), states, gelu_graph_counters(),
                )
            else:
                logger.info("[Benchmark] compile evidence stage={} config_use_compile={} transformer_states={}",
                            stage, config.get("use_compile", False), states)
            if compile_kind == "feature":
                feature_compile_evidence(stage)

        evidence("after_warmup")
        save_target = Path(args.save_result_path) if args.save_result_path else None
        for number in (1, 2):
            data = request_data.copy()
            if save_target is not None:
                data["save_result_path"] = str(save_target.with_name(f"request_{number}{save_target.suffix}"))
                data["return_result_tensor"] = False
            else:
                data.pop("save_result_path", None)
                data["return_result_tensor"] = True
            input_info = runner.prepare_request(data)
            print_request(input_info, runner.get_supported_request_fields(input_info.task))
            scheduler = getattr(runner, "scheduler", None)
            if scheduler is not None and hasattr(scheduler, "generator"):
                scheduler.generator = None
                logger.info("[Benchmark] reset scheduler generator for seed={}", input_info.seed)
            logger.info("[Benchmark] request {} begin rank={} seed={} save_result_path={}",
                        number, os.environ.get("RANK", "0"), input_info.seed, input_info.save_result_path)
            result = runner.run_request(input_info)
            del result
            logger.info("[Benchmark] request {} end rank={}", number, os.environ.get("RANK", "0"))
            evidence(f"after_request_{number}")
            if number == 1:
                infer.distributed_barrier()
                logger.info("[Benchmark] between-request distributed_barrier() returned")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group cleaned up")
