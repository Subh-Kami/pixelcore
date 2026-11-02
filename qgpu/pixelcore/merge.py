"""Compile the dedicated PixelCore into Qwen3.5-2B (or a standalone circuit model).

Weight emission and merge conventions adapt qwenGPU/qwenputer's MIT code. The
game circuit itself is in circuit.py; no SIMT GPU or external game engine is used.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from chipformer.q35 import merge as qm, tiny
from chipformer.q35.emit import Layout, allocate, embedding_rows, emit_layers
from chipformer.q35.ir import Target
from safetensors.torch import save_file

from qgpu.pixelcore.circuit import PixelCore
from qgpu.pixelcore.spec import TEXT, VOCAB, token_name

PREFIX = "model.language_model."


def tokenizer(src, dst, first):
    from tokenizers import AddedToken, Tokenizer

    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    n = tok.get_vocab_size(with_added_tokens=True)
    added = [AddedToken(f"<|reserved_{i}|>", special=True) for i in range(first - n)]
    added += [AddedToken(token_name(t), special=True) for t in range(VOCAB)]
    tok.add_special_tokens(added)
    assert tok.token_to_id(token_name(0)) == first
    tok.save(str(dst / "tokenizer.json"))
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    dec = cfg.setdefault("added_tokens_decoder", {})
    for i, t in enumerate(added):
        dec[str(n + i)] = {"content": t.content, "lstrip": False, "normalized": False,
                           "rstrip": False, "single_word": False, "special": True}
    (dst / "tokenizer_config.json").write_text(json.dumps(cfg))
    for name in ("chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json",
                 "generation_config.json", "LICENSE"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)


def build(src, dst):
    from chipformer.q35.mute import fit_anti_sink, mute

    src, dst = Path(src), Path(dst)
    cfg, sd = qm.load(src)
    tc = cfg["text_config"]
    d0, v0, inter = tc["hidden_size"], tc["vocab_size"], tc["intermediate_size"]
    circuit = PixelCore(Target(tc["linear_num_key_heads"], tc["linear_num_value_heads"], inter))
    keep = set(circuit.keep)
    alloc = allocate(circuit, None, keep, inter)
    hidden, ballast = qm.pick_hidden(d0, 1 + alloc.n_slots)
    layout = Layout(hidden=hidden, off=d0, m=ballast, slots=alloc.slots)
    print(f"PixelCore: {circuit.n_layers()} layers; {alloc.n_slots} slots; hidden {d0} → {hidden}", flush=True)
    cfg, sd = qm.widen(cfg, sd, hidden)
    tc = cfg["text_config"]
    n0, ncore = tc["num_hidden_layers"], circuit.n_layers()
    shift = re.compile(r"^(model\.language_model\.layers\.)(\d+)\.")
    sd = {shift.sub(lambda m: f"{m[1]}{int(m[2]) + ncore}.", k): v for k, v in sd.items()}
    tc["layer_types"] = ["linear_attention"] * ncore + tc["layer_types"]
    tc["num_hidden_layers"] += ncore
    vocab = -(-(v0 + VOCAB) // 64) * 64
    tc["vocab_size"] = vocab
    mute(sd, tc, n0, layout.dim(circuit.keep[-1]), hidden ** 0.5,
         sinks=fit_anti_sink(src, "cpu"), first=ncore)
    layers = emit_layers(circuit, layout, tc, 0, PREFIX, alloc, keep)
    sd.update({k: v.to(torch.bfloat16) for k, v in layers.items()})
    del layers
    full = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    full[:v0, :d0] = sd[PREFIX + "embed_tokens.weight"][:, :d0]
    text = embedding_rows(circuit, layout, [TEXT], hidden)[0].to(torch.bfloat16)
    full[:v0, d0:] = text[d0:]
    full[v0:v0 + VOCAB] = embedding_rows(circuit, layout, list(range(VOCAB)), hidden).to(torch.bfloat16)
    sd[PREFIX + "embed_tokens.weight"] = full
    bias = "model.visual.merger.linear_fc2.bias"
    if bias in sd:
        sd[bias] = sd[bias].clone()
        sd[bias][d0:] = text[d0:]
    norm = sd[PREFIX + "norm.weight"].float()
    norm[d0:] = -1
    for expr in circuit.code:
        norm[layout.dim(next(iter(expr.terms)))] = tiny.code_gain(layout)
    norm[layout.dim(next(iter(circuit.tau.terms)))] = qm.TAU_GAIN
    sd[PREFIX + "norm.weight"] = norm.to(torch.bfloat16)
    dst.mkdir(parents=True, exist_ok=True)
    shards, shard, size = [], {}, 0
    for k in sorted(sd):
        shard[k] = sd[k].contiguous()
        size += sd[k].numel() * 2
        if size > 2**31:
            shards.append(shard)
            shard, size = {}, 0
    if shard:
        shards.append(shard)
    index = {"metadata": {"total_size": sum(t.numel() * 2 for t in sd.values())}, "weight_map": {}}
    for i, sh in enumerate(shards):
        name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(dst / name), metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in sh})
    (dst / "model.safetensors.index.json").write_text(json.dumps(index))
    (dst / "config.json").write_text(json.dumps(cfg))
    tokenizer(src, dst, v0)
    meta = {"kind": "merged", "base": src.name, "core_layers": ncore,
            "base_layers": n0, "hidden": hidden, "slots": alloc.n_slots,
            "chip_token_offset": v0, "chip_vocab": VOCAB, "game": "WeightPong",
            "frame_tokens": 18, "tensor_bytes": index["metadata"]["total_size"]}
    (dst / "pixelcore.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


def build_tiny(dst):
    circuit = PixelCore()
    model = tiny.build(circuit, list(range(VOCAB)),
                       [next(iter(e.terms)) for e in circuit.code], share=True)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst)
    meta = {"kind": "standalone", "core_layers": circuit.n_layers(),
            "chip_token_offset": 0, "chip_vocab": VOCAB,
            "hidden": model.config.hidden_size, "game": "WeightPong"}
    (dst / "pixelcore.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--base", type=Path)
    args = parser.parse_args()
    if args.base:
        build(args.base, args.dst)
    else:
        build_tiny(args.dst)


if __name__ == "__main__":
    main()
