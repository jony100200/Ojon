"""Opt-in PyTorch inference bridge for Ojon's Comfy-Kitchen W4A8 checkpoints.

One runtime adapter, not another converter. This is experimental: the CPU eager
path is tested; fused GPU execution and Echo end-to-end inference are not.
The caller owns device placement and must load unquantised bias values first.
"""
from contextlib import ExitStack, nullcontext
import json

import torch
from torch import nn
from safetensors import safe_open
import comfy_kitchen as ck


_NO_BIAS_OVERRIDE = object()


class PackedW4A8Linear(nn.Module):
    """Inference-only replacement for one nn.Linear, retaining packed buffers."""

    def __init__(self, original, packed, s_rel, s_channel, codebook, *,
                 backend=None, bias_override=_NO_BIAS_OVERRIDE):
        super().__init__()
        if not isinstance(original, nn.Linear):
            raise TypeError("expected torch.nn.Linear")
        self.in_features = original.in_features
        self.out_features = original.out_features
        if tuple(packed.shape) != (self.out_features, self.in_features // 2):
            raise ValueError("packed W4 shape does not match nn.Linear")
        if tuple(s_rel.shape) != (self.out_features, self.in_features // 16):
            raise ValueError("W4 group-scale shape does not match nn.Linear")
        if tuple(s_channel.shape) != (self.out_features,) or tuple(codebook.shape) != (16,):
            raise ValueError("W4 channel-scale or codebook shape does not match nn.Linear")
        if packed.dtype != torch.int8 or s_rel.dtype != torch.float8_e4m3fn:
            raise TypeError("expected W4 packed int8 and FP8 E4M3 relative scales")
        if bias_override is _NO_BIAS_OVERRIDE:
            if original.weight.is_meta or (original.bias is not None and original.bias.is_meta):
                raise ValueError("materialise the original layer's bias before replacing it")
            bias = original.bias.detach().clone() if original.bias is not None else None
        else:
            bias = bias_override
            if (original.bias is None) != (bias is None):
                raise ValueError("W4A8 bias presence does not match nn.Linear")
            if bias is not None and tuple(bias.shape) != (self.out_features,):
                raise ValueError("W4A8 bias shape does not match nn.Linear")
        self.register_buffer("packed_weight", packed)
        self.register_buffer("weight_s_rel", s_rel)
        self.register_buffer("weight_s_channel", s_channel)
        self.register_buffer("weight_codebook", codebook)
        self.register_buffer("bias", bias)
        self.backend = backend

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError("W4 input feature dimension does not match")
        if x.device != self.packed_weight.device:
            raise RuntimeError("move the entire quantised model to the execution device first")
        flat = x.reshape(-1, self.in_features)
        use = ck.use_backend(self.backend) if self.backend is not None else nullcontext()
        with use:
            y = ck.tensor.w4a8_int8_linear(
                flat, self.packed_weight, self.weight_s_rel, self.weight_s_channel,
                codebook=self.weight_codebook, out_dtype=x.dtype,
            )
        if self.bias is not None:
            y = y + self.bias.to(dtype=y.dtype)
        return y.reshape(*x.shape[:-1], self.out_features)


def replace_w4a8_linears(model, checkpoint_path, *, layer_names=None, backend=None):
    """Replace exact-name nn.Linear modules from an Ojon W4A8 safetensors file.

    Validates all selected names/shapes before changing model modules. Packed
    tensors are read one layer at a time; unselected layers are untouched.
    Returns the names replaced. The model must already contain correct biases;
    this function does not make an original dense checkpoint load memory-free.
    """
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as ckpt:
        names = set(ckpt.keys())
        if layer_names is None:
            chosen = sorted(n.removesuffix(".comfy_quant") for n in names if n.endswith(".comfy_quant"))
        else:
            chosen = sorted(set(layer_names))
        if not chosen:
            raise ValueError("no W4A8 linear layers selected")
        checked = []
        for name in chosen:
            suffixes = (".comfy_quant", ".weight", ".weight_s_rel",
                        ".weight_s_channel", ".weight_codebook")
            if any(name + suffix not in names for suffix in suffixes):
                raise ValueError(f"incomplete W4A8 layer: {name}")
            config = json.loads(bytes(ckpt.get_tensor(name + ".comfy_quant").tolist()))
            if config.get("format") != "asym_w4a8_int8":
                raise ValueError(f"unsupported quantisation format in {name}")
            original = model.get_submodule(name)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"{name} is not nn.Linear")
            out_f, in_f = original.out_features, original.in_features
            if (ckpt.get_slice(name + ".weight").get_shape() != [out_f, in_f // 2]
                or ckpt.get_slice(name + ".weight_s_rel").get_shape() != [out_f, in_f // 16]
                or ckpt.get_slice(name + ".weight_s_channel").get_shape() != [out_f]
                or ckpt.get_slice(name + ".weight_codebook").get_shape() != [16]):
                raise ValueError(f"W4A8 packed shape mismatch: {name}")
            checked.append((name, original))
        for name, original in checked:
            module = PackedW4A8Linear(
                original,
                ckpt.get_tensor(name + ".weight"),
                ckpt.get_tensor(name + ".weight_s_rel"),
                ckpt.get_tensor(name + ".weight_s_channel"),
                ckpt.get_tensor(name + ".weight_codebook"),
                backend=backend,
            )
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, module)
        return chosen


def load_w4a8_meta_model(model, checkpoint_path, *, fallback_path=None,
                         dtype=torch.bfloat16, backend=None):
    """Stream an Ojon W4A8 checkpoint into an *unmaterialised* PyTorch model.

    Build ``model`` inside ``accelerate.init_empty_weights()`` first. Packed
    nn.Linear weights are never materialised densely; every remaining parameter
    is read a tensor at a time from the W4A8 file or optional base-model
    safetensors fallback. The W4A8 checkpoint wins when both contain a key.
    The original model is not modified on disk. Missing or incompatible weights
    fail *before* the first parameter is loaded; no random meta weights survive.
    Returns names of packed layers. CPU eager is the verified reference backend.
    """
    from accelerate.utils import set_module_tensor_to_device

    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    if not parameters or not all(t.is_meta for t in parameters.values()):
        raise ValueError("load_w4a8_meta_model requires a fully meta-initialised model")

    with ExitStack() as stack:
        quant = stack.enter_context(safe_open(str(checkpoint_path), framework="pt", device="cpu"))
        quant_names = set(quant.keys())
        base = (stack.enter_context(safe_open(str(fallback_path), framework="pt", device="cpu"))
                if fallback_path is not None else None)
        base_names = set(base.keys()) if base is not None else set()

        def source(name):
            if name in quant_names:
                return quant
            if name in base_names:
                return base
            return None

        chosen = sorted(n.removesuffix(".comfy_quant") for n in quant_names
                        if n.endswith(".comfy_quant"))
        if not chosen:
            raise ValueError("checkpoint contains no W4A8 linear layers")
        selected_weights = set()
        selected_biases = set()
        checked = []
        for name in chosen:
            original = model.get_submodule(name)
            if not isinstance(original, nn.Linear):
                raise TypeError(f"{name} is not nn.Linear")
            if not original.weight.is_meta:
                raise ValueError(f"{name} must be meta-initialised")
            suffixes = (".comfy_quant", ".weight", ".weight_s_rel",
                        ".weight_s_channel", ".weight_codebook")
            if any(name + suffix not in quant_names for suffix in suffixes):
                raise ValueError(f"incomplete W4A8 layer: {name}")
            config = json.loads(bytes(quant.get_tensor(name + ".comfy_quant").tolist()))
            if config.get("format") != "asym_w4a8_int8":
                raise ValueError(f"unsupported W4A8 format: {name}")
            out_f, in_f = original.out_features, original.in_features
            if (quant.get_slice(name + ".weight").get_shape() != [out_f, in_f // 2]
                or quant.get_slice(name + ".weight_s_rel").get_shape() != [out_f, in_f // 16]
                or quant.get_slice(name + ".weight_s_channel").get_shape() != [out_f]
                or quant.get_slice(name + ".weight_codebook").get_shape() != [16]):
                raise ValueError(f"W4A8 packed shape mismatch: {name}")
            bias_name = name + ".bias"
            if original.bias is not None:
                bias_file = source(bias_name)
                if bias_file is None or bias_file.get_slice(bias_name).get_shape() != [out_f]:
                    raise ValueError(f"missing or mismatched W4A8 bias: {bias_name}")
                selected_biases.add(bias_name)
            selected_weights.add(name + ".weight")
            checked.append((name, original))

        def can_pad_patch(name, got, wanted):
            return (name == "patch_embedding.weight" and len(got) == len(wanted) == 5
                    and got[0] == wanted[0] and 0 < got[1] < wanted[1]
                    and got[2:] == wanted[2:])

        to_load = []
        for name, tensor in {**parameters, **buffers}.items():
            if name in selected_weights or name in selected_biases:
                continue
            file = source(name)
            if file is None:
                if tensor.is_meta or name in parameters:
                    raise ValueError(f"no tensor for meta-model parameter: {name}")
                continue  # Non-meta constructor buffer, not in checkpoint.
            actual = file.get_slice(name).get_shape()
            expected = list(tensor.shape)
            if actual != expected and not can_pad_patch(name, actual, expected):
                raise ValueError(f"checkpoint shape mismatch for {name}: {actual} != {expected}")
            to_load.append((name, file, tensor, actual != expected))

        # All name and dimension checks passed. Never allocate a dense packed weight.
        for name, file, tensor, pad in to_load:
            value = file.get_tensor(name)
            if pad:
                expanded = torch.zeros(tuple(tensor.shape), dtype=value.dtype)
                expanded[:, :value.shape[1], ...] = value
                value = expanded
            target_dtype = dtype if tensor.is_floating_point() else tensor.dtype
            set_module_tensor_to_device(model, name, "cpu", value=value, dtype=target_dtype)

        for name, original in checked:
            bias_name = name + ".bias"
            bias_file = source(bias_name) if original.bias is not None else None
            bias = bias_file.get_tensor(bias_name).to(dtype) if bias_file is not None else None
            packed = PackedW4A8Linear(
                original, quant.get_tensor(name + ".weight"),
                quant.get_tensor(name + ".weight_s_rel"),
                quant.get_tensor(name + ".weight_s_channel"),
                quant.get_tensor(name + ".weight_codebook"),
                backend=backend, bias_override=bias,
            )
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, packed)

    remaining = [n for n, p in model.named_parameters() if p.is_meta]
    remaining += [n for n, b in model.named_buffers() if b.is_meta]
    if remaining:
        raise RuntimeError(f"unmaterialised tensors after W4A8 loading: {remaining[:5]}")
    return chosen
