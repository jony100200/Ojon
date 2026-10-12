"""Bundled adapter for public HQQ APIs. Invoked by Ojon via python -c.
No installed Python package, model or tokenizer is fetched automatically.
HQQ native qmodel.pt is NOT a GGUF file.
"""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Ojon local HQQ conversion bridge")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bits", type=int, choices=[2, 3, 4]
    parser.add_argument("--group", type=int, choices=[32, 64, 128])
    parser.add_argument("--device", choices=["cpu", "cuda", "xpu"])
    args = parser.parse_args()

    # Check source is strictly local even if someone invokes this bridge directly.
    source = Path(args.source).resolve(strict=True)
    if not (source / "config.json").is_file():
        raise ValueError("local source requires config.json")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("output exists")
    if source == output or source in output.parents:
        raise ValueError("output must be outside model source")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from hqq.core.quantize import BaseQuantizeConfig
    from hqq.models.hf.base import AutoHQQHFModel

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend not available")
    if args.device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("Intel torch.xpu not available")
    # The upstream HQQ native API supports 1, 2, 3, 4, and 8-bit weights.
    if args.bits not in [2, 3, 4]:
        raise ValueError("This Ojon bridge currently supports HQQ 2/3/4-bit only")

    compute_dtype = torch.float32 if args.device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False, torch_dtype="auto"
    )
    config = BaseQuantizeConfig(nbits=args.bits, group_size=args.group, axis=1)
    AutoHQQHFModel.quantize_model(
        model, quant_config=config, compute_dtype=compute_dtype, device=args.device
    )
    output.mkdir(parents=False, exist_ok=False)
    AutoHQQHFModel.save_quantized(model, str(output))
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=False
    )
    tokenizer.save_pretrained(str(output))
    if not (output / "qmodel.pt").is_file() or not (output / "config.json").is_file():
        raise RuntimeError("HQQ archive or configuration missing")


if __name__ == "__main__":
    main()
