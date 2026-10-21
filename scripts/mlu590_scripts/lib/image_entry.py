"""Run one MLU image benchmark with warmup and two requests per process."""

import argparse
import importlib
import os
from pathlib import Path

import torch
import torch._dynamo
import torch.distributed as dist
from loguru import logger

from lightx2v.infer import distributed_barrier
from lightx2v.models.runners.runner_factory import RUNNER_MODULES, build_runner
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.set_config import build_cli_inputs, init_parallel, print_config
from lightx2v.utils.utils import seed_all, validate_config_paths
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


ACTIVATIONS = {
    "qwen_image": ("lightx2v.models.networks.qwen_image.infer.transformer_infer", "gelu"),
    "longcat_image": ("lightx2v.models.networks.longcat_image.infer.transformer_infer", "gelu"),
    "flux2": ("lightx2v.models.networks.flux2.infer.transformer_infer", "silu"),
    "z_image": ("lightx2v.models.networks.z_image.infer.transformer_infer", "silu"),
}


def install_compile(config):
    model_cls = config["model_cls"]
    if model_cls not in ACTIVATIONS or not config.get("use_compile", False):
        raise ValueError(f"Compiled image benchmark is unavailable for {model_cls}")
    if model_cls == "qwen_image":
        if config.get("use_magi_compile", False) or config.get("cpu_offload", False):
            raise ValueError("Qwen GELU-only compile requires Magi and CPU offload disabled")
        if config.get("dit_quant_scheme", "Default") != "Default":
            raise ValueError("Qwen GELU-only compile requires unquantized weights")
        if torch._dynamo.config.suppress_errors:
            raise ValueError("Qwen GELU-only compile requires Dynamo suppress_errors=False")

    module_name, activation_name = ACTIVATIONS[model_cls]
    module = importlib.import_module(module_name)
    functional = torch.nn.functional
    original = getattr(functional, activation_name)
    state = {
        "model": model_cls,
        "activation": activation_name,
        "attempts": 0,
        "completed": 0,
        "graphs": 0,
        "executions": 0,
        "eager_blocks": 0,
        "dtypes": set(),
    }

    def activation(tensor):
        if activation_name == "gelu":
            return original(tensor, approximate="tanh")
        return original(tensor)

    def backend(graph_module, example_inputs):
        from torch._dynamo.backends.registry import lookup_backend

        operations = tuple(str(node.target) for node in graph_module.graph.nodes)
        if not any(activation_name in operation.lower() for operation in operations):
            raise RuntimeError(f"Compiled graph lacks {activation_name}: {operations}")
        if model_cls == "qwen_image":
            calls = [node for node in graph_module.graph.nodes
                     if node.op in {"call_function", "call_method"}]
            if len(calls) != 1 or "gelu" not in str(calls[0].target).lower():
                raise RuntimeError(f"Qwen GELU graph has unexpected operations: {operations}")
        tensors = [item for item in example_inputs if isinstance(item, torch.Tensor)]
        if not tensors or any(item.device.type != "mlu" for item in tensors):
            raise RuntimeError("Compiled activation did not receive MLU tensors")
        compiled = lookup_backend("inductor")(graph_module, example_inputs)
        state["graphs"] += 1
        logger.info("[Compile] graph={} model={} activation={} operations={}",
                    state["graphs"], model_cls, activation_name, operations)

        def counted(*args, **kwargs):
            result = compiled(*args, **kwargs)
            state["executions"] += 1
            return result

        counted._boxed_call = getattr(compiled, "_boxed_call", False)
        return counted

    compiled_activation = torch.compile(activation, backend=backend, dynamic=None, fullgraph=True)

    def apply_compiled(tensor, *, approximate=None, inplace=False):
        if activation_name == "gelu" and approximate != "tanh":
            return original(tensor, approximate=approximate or "none")
        if activation_name == "silu" and inplace:
            return original(tensor, inplace=True)
        if tensor.device.type != "mlu":
            return activation(tensor)
        state["attempts"] += 1
        state["dtypes"].add(str(tensor.dtype))
        result = compiled_activation(tensor)
        state["completed"] += 1
        return result

    class FunctionalProxy:
        def __getattr__(self, name):
            return getattr(functional, name)

    proxy = FunctionalProxy()
    setattr(proxy, activation_name, apply_compiled)
    module.F = proxy

    if model_cls == "qwen_image":
        from lightx2v.models.networks.qwen_image.infer.transformer_infer import QwenImageTransformerInfer

        def eager_block(self, block_idx, block, *args):
            state["eager_blocks"] += 1
            return self.infer_block(block, *args)

        QwenImageTransformerInfer.run_block = eager_block
        logger.info("[Compile] Qwen block dispatch is eager; image/text GELU is compiled")
    else:
        logger.info("[Compile] {} FFN activation {} is compiled", model_cls, activation_name)
    return state


def compile_evidence(state, stage, previous):
    current = {key: state[key] for key in ("attempts", "completed", "graphs", "executions", "eager_blocks")}
    if state["model"] == "qwen_image":
        logger.info("[GELU-only] evidence stage={} states={}", stage, [{
            "backend_graphs": current["graphs"],
            "graph_executions": current["executions"],
            "gelu_attempts": current["attempts"],
            "gelu_completed": current["completed"],
            "eager_block_calls": current["eager_blocks"],
        }])
    else:
        logger.info(
            "[FeatureCompile] evidence stage={} model={} activation={} attempts={} completed={} graphs={} mlu_dtypes={}",
            stage, state["model"], state["activation"], current["attempts"],
            current["completed"], current["graphs"], sorted(state["dtypes"]),
        )
    if current["graphs"] < 1 or current["attempts"] < 1:
        raise RuntimeError(f"No compiled activation was observed at {stage}")
    if current["attempts"] != current["completed"]:
        raise RuntimeError(f"Compiled activation did not complete at {stage}")
    if current["executions"] < current["completed"]:
        raise RuntimeError(f"Compiled graph execution evidence is missing at {stage}")
    if state["model"] == "qwen_image" and current["eager_blocks"] < 1:
        raise RuntimeError(f"Qwen eager block/GELU compile path was not exercised at {stage}")
    if previous is not None:
        if current["attempts"] <= previous["attempts"]:
            raise RuntimeError(f"Compiled activation was not used during {stage}")
        if current["executions"] <= previous["executions"]:
            raise RuntimeError(f"Compiled graph did not execute during {stage}")
    return current


def install_warmup(config, request_data):
    if config["model_cls"] == "qwen_image":
        logger.info("[Warmup] using QwenImageRunner.run_warmup")
        return

    model_cls = config["model_cls"]
    runner_module = importlib.import_module(RUNNER_MODULES[model_cls])
    runner_name = {
        "longcat_image": "LongCatImageRunner",
        "flux2": "Flux2Runner",
        "z_image": "ZImageRunner",
    }[model_cls]
    runner_class = getattr(runner_module, runner_name)

    if model_cls == "longcat_image":
        original_run_pipeline = runner_class.run_pipeline

        def profiled_run_pipeline(self, input_info):
            with ProfilingContext4DebugL1("RUN pipeline"):
                return original_run_pipeline(self, input_info)

        runner_class.run_pipeline = profiled_run_pipeline

    def run_warmup(self):
        data = request_data.copy()
        data["save_result_path"] = None
        data["return_result_tensor"] = True
        info = self.prepare_request(data)
        seed_all(info.seed)
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "generator"):
            scheduler.generator = None
        logger.info("[Benchmark] warmup begin model={} seed={} save_result_path=None", model_cls, info.seed)
        result = self.run_pipeline(info)
        del result
        logger.info("[Benchmark] warmup end model={}", model_cls)

    runner_class.run_warmup = run_warmup
    logger.info("[Warmup] installed full-pipeline warmup for {}", runner_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", choices=tuple(ACTIVATIONS), required=True)
    parser.add_argument("--model-variant", dest="model_variant")
    parser.add_argument("--task", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--aspect_ratio")
    parser.add_argument("--size", type=int, nargs=2, required=True)
    parser.add_argument("--return_result_tensor", action="store_true", default=None)
    parser.add_argument("--save_result_path")
    args = parser.parse_args()
    seed_all(args.seed)
    warmup = args.warmup
    del args.warmup

    config, request_data = build_cli_inputs(args)
    config["warmup"] = warmup
    if config["parallel"]:
        platform_device = PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"))
        platform_device.init_parallel_env()
        init_parallel(config)
    print_config(config, title="Startup config")
    validate_config_paths(config)

    if not config.get("enable_cfg", False) and "negative_prompt" in request_data:
        request_data.pop("negative_prompt")

    importlib.import_module("lightx2v.common.ops")
    compile_state = install_compile(config)
    if warmup:
        install_warmup(config, request_data)

    with ProfilingContext4DebugL1("Total Cost"):
        runner = build_runner(config)
        logger.info("[Warmup] enabled={} completed={}", warmup, runner._warmup_done)
        previous = compile_evidence(compile_state, "after_warmup", None)
        save_target = Path(request_data["save_result_path"]) if request_data.get("save_result_path") else None
        for number in (1, 2):
            data = request_data.copy()
            if save_target is not None:
                data["save_result_path"] = str(save_target.with_name(f"request_{number}{save_target.suffix}"))
            info = runner.prepare_request(data)
            scheduler = getattr(runner, "scheduler", None)
            if scheduler is not None and hasattr(scheduler, "generator"):
                scheduler.generator = None
            logger.info("[Benchmark] request {} begin rank={} seed={} save_result_path={}",
                        number, os.getenv("RANK", "0"), info.seed, info.save_result_path)
            result = runner.run_request(info)
            del result
            logger.info("[Benchmark] request {} end rank={}", number, os.getenv("RANK", "0"))
            previous = compile_evidence(compile_state, f"after_request_{number}", previous)
            if number == 1:
                distributed_barrier()
                logger.info("[Benchmark] between-request distributed_barrier() returned")

    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group cleaned up")


if __name__ == "__main__":
    main()
