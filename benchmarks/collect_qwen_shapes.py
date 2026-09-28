#!/usr/bin/env python3
"""Actual Qwen3 forward hooks under controlled token workloads."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_hgemm.provenance import code_fingerprint

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/lq/vllm-1p1d-v029/model/Qwen3-4B")
    p.add_argument("--output", type=Path, default=Path("configs/qwen3_shapes.csv"))
    p.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    p.add_argument("--lengths", nargs="+", type=int, default=[128, 512])
    p.add_argument("--decode-steps", type=int, default=3)
    p.add_argument("--modules", nargs="+",
                   default=["q_proj", "o_proj", "gate_proj", "down_proj"])
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()
    if min(args.batches + args.lengths + [args.decode_steps]) <= 0:
        raise ValueError("workload dimensions must be positive")
    model_path = Path(args.model).expanduser()
    is_local_model = model_path.is_dir()
    if model_path.is_absolute() and not is_local_model:
        raise FileNotFoundError(f"local model directory not found: {model_path}")
    if is_local_model:
        model_path = model_path.resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"missing config.json: {model_path}")
        index_path = model_path / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            shards = set(index.get("weight_map", {}).values())
            if not shards:
                raise ValueError(f"empty safetensors weight map: {index_path}")
            missing = [name for name in sorted(shards)
                       if not (model_path / name).is_file()
                       or (model_path / name).stat().st_size == 0]
            if missing:
                raise FileNotFoundError(f"missing/empty model shards: {missing}")
        args.model = str(model_path)
    local_files_only = args.local_files_only or is_local_model
    print(f"Model: {args.model}; local_files_only={local_files_only}; dtype=FP16")
    torch.manual_seed(2026)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa",
        local_files_only=local_files_only).to("cuda").eval()
    counts = Counter()
    state = {}
    handles = []
    def hook_for(name):
        def hook(module, inputs, output):
            x = inputs[0]
            counts[(state["phase"], state["batch"], state["length"], name,
                    x.numel() // x.shape[-1], module.out_features,
                    module.in_features, json.dumps(list(x.shape)))] += 1
        return hook
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in args.modules:
            handles.append(module.register_forward_hook(hook_for(name)))
    if not handles:
        raise ValueError("no matching Linear modules")
    try:
        for batch in args.batches:
            for length in args.lengths:
                state.update(phase="prefill", batch=batch, length=length)
                ids = torch.randint(0, model.config.vocab_size,
                                    (batch, length), device="cuda")
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                               use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                state["phase"] = "decode"
                for step in range(args.decode_steps):
                    token = torch.randint(0, model.config.vocab_size,
                                          (batch, 1), device="cuda")
                    mask = torch.ones((batch, length + step + 1),
                                      device="cuda", dtype=torch.long)
                    output = model(input_ids=token, attention_mask=mask,
                                   past_key_values=cache, use_cache=True,
                                   logits_to_keep=1)
                    cache = output.past_key_values
                del output, cache
    finally:
        for handle in handles:
            handle.remove()
    fields = ["phase", "batch", "prompt_length", "module",
              "M", "N", "K", "input_shape", "calls"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, count in counts.items():
            writer.writerow(dict(zip(fields, (*key, count))))
    args.output.with_suffix(".manifest.json").write_text(json.dumps({
        "model": args.model, "load_dtype": "float16",
        "local_files_only": local_files_only, "model_commit": getattr(model.config, "_commit_hash", None),
        "workload": "controlled random token inputs, actual model forward",
        "batches": args.batches, "lengths": args.lengths, "modules": args.modules,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "code_fingerprint": code_fingerprint(),
    }, indent=2))
    print(f"{len(counts)} hook records -> {args.output}")

if __name__ == "__main__":
    main()
