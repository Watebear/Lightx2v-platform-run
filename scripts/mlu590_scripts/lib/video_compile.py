"""MLU video activation compilation used by the two-request benchmarks."""

import importlib

import torch
from loguru import logger


_LTX_GEMMA_ATTENTION_STATE = None
_FEATURE_COMPILE_STATE = None


def install_wan_gelu_only_compile(config):
    """Compile Wan's FFN GELU while leaving attention and matmuls eager.

    This path supports the MLU BF16 cases without CPU offload.
    """
    if config["model_cls"] not in {"wan2.1", "wan2.2_moe"}:
        raise ValueError("GELU compilation supports wan2.1 and wan2.2_moe only")
    if not config.get("use_compile", False):
        raise ValueError("GELU compilation requires use_compile=true")
    if config.get("cpu_offload", False):
        raise ValueError("GELU compilation requires cpu_offload=false")
    if config.get("feature_caching", "NoCaching") != "NoCaching":
        raise ValueError("GELU compilation requires NoCaching")

    from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
    from torch._dynamo.backends.registry import lookup_backend

    def eager_block(self, block_idx, block, *args):
        self.block_idx = block_idx
        return self.infer_block(block, *args)

    def gelu_only(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa=None):
        # The tested BF16 cases take the ordinary FFN branch.
        if self.clean_cuda_cache or self._use_mxfp8_quant_fuse():
            raise RuntimeError("GELU compilation requires clean_cuda_cache=false and no MXFP8 fusion")

        x.add_(attn_out)
        if hasattr(phase, "smooth_norm2_weight"):
            norm2_weight = (1 + c_scale_msa.squeeze()) * phase.smooth_norm2_weight.tensor
            norm2_bias = c_shift_msa.squeeze() * phase.smooth_norm2_bias.tensor
            norm2_out = phase.norm2.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm2_out = norm2_out.to(self.sensitive_layer_dtype)
            norm2_out.mul_(norm2_weight).add_(norm2_bias)
        else:
            norm2_out = phase.norm2.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm2_out = norm2_out.to(self.sensitive_layer_dtype)
            if self._can_use_mxfp8_modulate_quant(norm2_out, c_scale_msa, c_shift_msa):
                raise RuntimeError("GELU compilation does not support MXFP8 modulate quantization")
            norm2_out = self.modulate_func(norm2_out, scale=c_scale_msa, shift=c_shift_msa).squeeze()

        if self.sensitive_layer_dtype != self.infer_dtype:
            norm2_out = norm2_out.to(self.infer_dtype)

        cache = self.__dict__.setdefault("_benchmark_gelu_compiled", {})
        entry = cache.get("gelu")
        if entry is None:
            entry = {"compiled": None, "attempts": 0, "completed": 0, "inductor_graphs": 0}

            def gelu_core(tensor):
                return torch.nn.functional.gelu(tensor, approximate="tanh")

            def checked_inductor_backend(graph_module, example_inputs):
                operations = tuple(str(node.target) for node in graph_module.graph.nodes)
                if not any("gelu" in operation.lower() for operation in operations):
                    raise RuntimeError(f"Fullgraph FX graph lacks GELU: {operations}")
                tensor_inputs = [value for value in example_inputs if isinstance(value, torch.Tensor)]
                if not tensor_inputs or any(value.device.type != "mlu" for value in tensor_inputs):
                    raise RuntimeError("GELU Inductor graph did not receive MLU tensor inputs")
                input_specs = tuple((tuple(value.shape), str(value.dtype), str(value.device)) for value in tensor_inputs)
                compiled = lookup_backend("inductor")(graph_module, example_inputs)
                entry["inductor_graphs"] += 1
                logger.info(
                    "[BenchmarkGELU] Inductor compiled GELU fullgraph graph_number={} operations={} inputs={}",
                    entry["inductor_graphs"], operations, input_specs,
                )
                return compiled

            entry["compiled"] = torch.compile(
                gelu_core, backend=checked_inductor_backend, dynamic=None, fullgraph=True,
            )
            cache["gelu"] = entry
            logger.info("[BenchmarkGELU] created fullgraph Inductor GELU wrapper")

        y = phase.ffn_0.apply(norm2_out)
        entry["attempts"] += 1
        y = entry["compiled"](y)
        entry["completed"] += 1
        return phase.ffn_2.apply(y)

    WanTransformerInfer.run_block = eager_block
    WanTransformerInfer.infer_ffn = gelu_only
    logger.info("[BenchmarkGELU] installed eager Wan attention/matmuls + fullgraph Inductor GELU for {}", config["model_cls"])



def gelu_graph_counters():
    """Return compiler counters without depending on a specific torch build."""
    result = {}
    try:
        from torch._dynamo.utils import counters

        result["unique_graphs"] = counters.get("stats", {}).get("unique_graphs", 0)
        result["captured_calls"] = counters.get("stats", {}).get("calls_captured", 0)
        result["graph_breaks"] = sum(counters.get("graph_break", {}).values())
    except Exception as exc:
        result["dynamo_counter_error"] = repr(exc)
    try:
        from torch._inductor import metrics

        result["generated_kernels"] = metrics.generated_kernel_count
    except Exception as exc:
        result["inductor_metric_error"] = repr(exc)
    return result



def install_activation_compile(config):
    """Compile the FFN activation for video models without block compilation.

    These models do not execute the project's whole-block compile path (or,
    for LTX2, its large offloaded block cannot use this compile path).
    Replace only the activation used by the original FFN.  Every compiled
    call is checked on MLU tensors and counted through both requests.
    """
    global _FEATURE_COMPILE_STATE
    model_cls = config["model_cls"]
    mapping = {
        "hunyuan_video_1.5": ("lightx2v.models.networks.hunyuan_video.infer.transformer_infer", "gelu"),
        "wan2.1_sf": (None, "gelu"),
        "ltx2": (None, "gelu"),
    }
    if model_cls not in mapping:
        raise ValueError(f"No activation compile integration for {model_cls}")
    if not config.get("use_compile", False):
        raise ValueError("Activation compile case requires use_compile=true")

    module_name, activation_name = mapping[model_cls]
    functional = torch.nn.functional
    original_activation = getattr(functional, activation_name)
    state = {"model_cls": model_cls, "activation": activation_name,
             "attempts": 0, "completed": 0, "inductor_graphs": 0,
             "graph_breaks": 0, "mlu_dtypes": set()}

    def core(tensor):
        return original_activation(tensor, approximate="tanh")

    def checked_inductor_backend(graph_module, example_inputs):
        from torch._dynamo.backends.registry import lookup_backend
        operations = tuple(str(node.target) for node in graph_module.graph.nodes)
        if not any(activation_name in operation.lower() for operation in operations):
            raise RuntimeError(f"Compiled graph lacks {activation_name}: {operations}")
        tensors = [value for value in example_inputs if isinstance(value, torch.Tensor)]
        if not tensors or any(value.device.type != "mlu" for value in tensors):
            raise RuntimeError(f"Compiled {activation_name} graph has non-MLU inputs")
        compiled = lookup_backend("inductor")(graph_module, example_inputs)
        state["inductor_graphs"] += 1
        specs = tuple((tuple(value.shape), str(value.dtype), str(value.device)) for value in tensors)
        logger.info("[FeatureCompile] Inductor fullgraph model={} activation={} graph={} operations={} inputs={}",
                    model_cls, activation_name, state["inductor_graphs"], operations, specs)
        return compiled

    compiled_activation = torch.compile(core, backend=checked_inductor_backend,
                                        dynamic=None, fullgraph=True)

    def apply_compiled(tensor, *, approximate=None, inplace=False):
        if approximate != "tanh":
            return original_activation(tensor, approximate=approximate or "none")
        if tensor.device.type != "mlu":
            return original_activation(tensor, approximate="tanh")
        state["attempts"] += 1
        state["mlu_dtypes"].add(str(tensor.dtype))
        result = compiled_activation(tensor)
        state["completed"] += 1
        return result

    if module_name:
        module = importlib.import_module(module_name)
        class FunctionalProxy:
            def __getattr__(self, name):
                return getattr(functional, name)

        proxy = FunctionalProxy()
        setattr(proxy, activation_name, apply_compiled)
        module.F = proxy
        logger.info("[FeatureCompile] installed module-local {} proxy for {}", activation_name, model_cls)
    elif model_cls == "wan2.1_sf":
        from lightx2v.models.networks.wan.infer.self_forcing.transformer_infer import WanSFTransformerInfer

        def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa=None):
            if self.clean_cuda_cache:
                raise RuntimeError("WanSF activation compile requires clean_cuda_cache=false")
            x.add_(attn_out)
            num_frames = c_shift_msa.shape[0]
            frame_seqlen = x.shape[0] // num_frames
            if hasattr(phase, "smooth_norm2_weight"):
                norm2_weight = (1 + c_scale_msa.squeeze()) * phase.smooth_norm2_weight.tensor
                norm2_bias = c_shift_msa.squeeze() * phase.smooth_norm2_bias.tensor
            else:
                norm2_weight = 1 + c_scale_msa
                norm2_bias = c_shift_msa
            norm2_out = phase.norm2.apply(x)
            norm2_out = norm2_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
            norm2_out.mul_(norm2_weight).add_(norm2_bias)
            norm2_out = norm2_out.flatten(0, 1)
            y = phase.ffn_0.apply(norm2_out)
            y = apply_compiled(y, approximate="tanh")
            return phase.ffn_2.apply(y)

        WanSFTransformerInfer.infer_ffn = infer_ffn
        logger.info("[FeatureCompile] installed WanSF FFN GELU path")
    else:
        from lightx2v.models.networks.ltx2.infer.transformer_infer import LTX2TransformerInfer

        def local_init_compile(self, _config):
            self.use_compile = False
            self.compiled_blocks = {}
            logger.info("[FeatureCompile] LTX2 whole-block cache disabled; FFN GELU local compile active")

        def infer_ffn(self, ffn_phase, x):
            x = ffn_phase.net_0_proj.apply(x)
            x = apply_compiled(x, approximate="tanh")
            return ffn_phase.net_2.apply(x)

        LTX2TransformerInfer.init_compile = local_init_compile
        LTX2TransformerInfer._infer_ffn = infer_ffn
        logger.info("[FeatureCompile] installed LTX2 FFN GELU path")
    _FEATURE_COMPILE_STATE = state



def feature_compile_evidence(stage):
    state = _FEATURE_COMPILE_STATE
    if state is None:
        raise RuntimeError("Feature compile state not installed")
    logger.info("[FeatureCompile] evidence stage={} model={} activation={} attempts={} completed={} graphs={} mlu_dtypes={}",
                stage, state["model_cls"], state["activation"], state["attempts"],
                state["completed"], state["inductor_graphs"], sorted(state["mlu_dtypes"]))



def register_ltx_gemma_attention_alias(config):
    """Keep the MLU Gemma kernel while avoiding Transformers 5 flash-name lookup."""
    global _LTX_GEMMA_ATTENTION_STATE
    alias = "mlu_attn_local"
    if config.get("model_cls") != "ltx2" or config.get("gemma_attn_implementation") != {"text_config": alias}:
        return

    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from lightx2v.models.input_encoders.hf.ltx2.gemma.model import (
        mlu_flash_attention_forward,
        register_gemma_attention_backend,
    )

    # This resolves the platform op before any model weights are loaded. The
    # callback below still calls the project's original torch_mlu_ops path.
    register_gemma_attention_backend("mlu_flash_attn")
    _LTX_GEMMA_ATTENTION_STATE = {"alias": alias, "calls": 0}

    def mlu_attn_local_forward(*args, **kwargs):
        _LTX_GEMMA_ATTENTION_STATE["calls"] += 1
        return mlu_flash_attention_forward(*args, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register(alias, mlu_attn_local_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(alias, sdpa_mask)
    logger.info(
        "[Benchmark] LTX2 Gemma text attention alias={} callback={} platform_op=mlu_flash_attn; image tower keeps its default",
        alias,
        mlu_flash_attention_forward.__qualname__,
    )

