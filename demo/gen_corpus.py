#!/usr/bin/env python3
"""Generate the scrolling-demo corpora for pii_duel.py.

The document is a readable, PII-sparse prose paragraph (demo/document.txt) tiled
to the target token count -- the same tile-to-length method the benchmark uses
(scripts/bench_torch.py) -- so "tokens" on screen means the same thing as in the
README. One tile = 160 tokens (pf-cli --tok-batch), giving an exact char<->token
mapping. Entity spans are the REAL pf-cli --classify output for the tile,
replicated per tile (each tile is identical, so the model finds the same spans).

Writes demo/traces/<scene>/{content.json,engines.json} for scene in cpu,gpu.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = open(HERE / "document.txt").read().rstrip("\n") + "\n\n"   # 731 chars
TOKENS_PER_TILE = 160                                             # pf-cli --tok-batch

# Real spans from: pf-cli --classify <model> 0.5 <<< "<tile>"  (offsets into SEED).
SEED_ENTITIES = [
    {"type": "person", "start": 236, "end": 244},   # John Doe
    {"type": "phone",  "start": 419, "end": 430},   # +1 555-0112
    {"type": "date",   "start": 531, "end": 541},   # 2026-05-12
    {"type": "email",  "start": 624, "end": 653},   # jane.roe@northside-clinic.org
]

L = len(SEED)


def build(target_tokens):
    tiles = round(target_tokens / TOKENS_PER_TILE)
    doc = SEED * tiles
    ents = []
    for t in range(tiles):
        off = t * L
        for e in SEED_ENTITIES:
            ents.append({"type": e["type"], "start": e["start"] + off, "end": e["end"] + off})
    return doc, ents, tiles * TOKENS_PER_TILE


def write_scene(scene, target_tokens, note, engines):
    doc, ents, n_tokens = build(target_tokens)
    d = Path(__file__).resolve().parent / "traces" / scene
    d.mkdir(parents=True, exist_ok=True)
    content = {"document": doc, "n_tokens": n_tokens, "note": note, "entities": ents}
    json.dump(content, open(d / "content.json", "w"), ensure_ascii=False)
    json.dump(engines, open(d / "engines.json", "w"), indent=2, ensure_ascii=False)
    chars = len(doc)
    print(f"{scene}: {len(ents):,} entities, {n_tokens:,} tokens, {chars:,} chars "
          f"({chars/n_tokens:.2f} chars/tok)")
    for e in engines:
        if e.get("oom_at_tokens"):
            print(f"   {e['label']:<20} OOM at {e['oom_at_tokens']:,} tok "
                  f"(~{e['oom_at_tokens']/e['tps']:.2f}s) @ {e['tps']:,} tok/s")
        else:
            print(f"   {e['label']:<20} {n_tokens/e['tps']:.2f}s @ {e['tps']:,} tok/s")


# CPU: 8k doc, both finish (README CPU 8 192 row: ours 2 332, HF 304 tok/s).
write_scene("cpu", 8192, "8k-token document · CPU (Ryzen 9 7900)", [
    {"key": "ours", "label": "privacy-filter.cpp", "device": "CPU", "tps": 2332, "hero": True},
    {"key": "hf",   "label": "HF Transformers",    "device": "CPU", "tps": 304,  "hero": False},
])

# GPU: 132k doc. Ours (Vulkan) runs the whole thing (README 131 072 row: 81 105
# tok/s). HF (CUDA) OOMs past ~16k on a 16 GiB GPU -> dies at 16 384 tokens; use
# its measured 8k throughput (14 154 tok/s) as the rate up to the wall.
write_scene("gpu", 131072, "132k-token document · GPU (RTX 5070 Ti, 16 GiB)", [
    {"key": "ours", "label": "privacy-filter.cpp", "device": "GPU · Vulkan", "tps": 81105, "hero": True},
    {"key": "hf",   "label": "HF Transformers",    "device": "GPU · CUDA",   "tps": 14154,
     "hero": False, "oom_at_tokens": 16384},
])
