# privacy-filter.cpp

Minimal [GGML](https://github.com/ggml-org/ggml) inference engine for the
`openai-privacy-filter` token-classification model family
([openai/privacy-filter](https://huggingface.co/openai/privacy-filter),
[OpenMed/privacy-filter-multilingual](https://huggingface.co/OpenMed/privacy-filter-multilingual)):
PII/NER entity spans with exact UTF-8 byte offsets. Stock upstream ggml — no
patches; the model's YaRN `truncate=false` frequencies are computed at load
time and fed to `ggml_rope_ext` as `freq_factors`.

Pre-converted GGUFs (arch `openai-privacy-filter`):
[`LocalAI-io/privacy-filter-multilingual-GGUF`](https://huggingface.co/LocalAI-io/privacy-filter-multilingual-GGUF)
and [`LocalAI-io/privacy-filter-GGUF`](https://huggingface.co/LocalAI-io/privacy-filter-GGUF).
Convert your own from a HF checkpoint with
[`scripts/convert.py`](scripts/convert.py) — self-contained, no llama.cpp
dependency (see [Convert](#convert)).

## Bench

A "redaction race" against stock HF Transformers on the same hardware:

**CPU — 8k-token document, real time.** Both finish; ours is 7.7× faster.

![CPU redaction race: privacy-filter.cpp vs HF Transformers on an 8k-token document](demo/out/pii_duel_cpu.gif)

**GPU — 132k-token document (4× slow-mo).** Ours runs flat to 131k tokens; HF
hits the 16 GiB memory wall and OOMs at ~16k.

![GPU redaction race: privacy-filter.cpp runs to 131k tokens while HF OOMs](demo/out/pii_duel_gpu.gif)

Full-quality MP4s: [CPU](demo/out/pii_duel_cpu_final.mp4) · [GPU](demo/out/pii_duel_gpu_final.mp4).

Single forward-pass latency and throughput vs stock HF Transformers (transformers
5.9, eager), Ryzen 9 7900 (12 threads) + RTX 5070 Ti, f16/fp16, matched token
counts ([scripts/bench_torch.py](scripts/bench_torch.py)). `tokens` is the input
sequence length classified in one forward pass (the whole document at once, not
generation); latency is `tokens ÷ tok/s`.

GPU — ours (Vulkan) vs HF (CUDA):

| tokens | HF (tok/s) | HF (ms) | ours (tok/s) | ours (ms) | speedup |
|-------:|-----------:|--------:|-------------:|----------:|--------:|
|    512 |      5 526 |      93 |      100 503 |         5 |     18× |
|  2 048 |     16 427 |     125 |      145 481 |        14 |    8.9× |
|  8 192 |     14 154 |     579 |      105 034 |        78 |    7.4× |
| 32 768 |        OOM |     OOM |       83 519 |       392 |       — |
| 131072 |        OOM |     OOM |       81 105 |     1 616 |       — |

CPU — ours vs HF (fp32):

| tokens | HF (tok/s) | HF (s) | ours (tok/s) | ours (s) | speedup |
|-------:|-----------:|-------:|-------------:|---------:|--------:|
|    512 |      2 171 |   0.24 |        3 564 |     0.14 |    1.6× |
|  2 048 |        978 |   2.09 |        3 490 |     0.59 |    3.6× |
|  8 192 |        304 |  26.95 |        2 332 |     3.51 |    7.7× |

The speedup widens with length because HF's full self-attention is O(n²) while
ours is banded/near-linear, so our tok/s stays roughly flat as HF's collapses.
Memory is flat ~2.8 GiB VRAM on a 16
GiB GPU. `release-portable` runtime-dispatches the best ggml-cpu ISA (AVX-512
without `-march=native`); flash + banded attention default on. See
[docs/cpu-perf.md](docs/cpu-perf.md).

Reproduce the numbers:

```sh
cmake --preset release-portable && cmake --build --preset release-portable -j
build/release-portable/bin/pf-bench model.gguf [cpu|vulkan] [iters] [lengths]
```

## Build

```sh
git clone --recursive <repo>
cmake --preset release && cmake --build --preset release -j
```

Presets: `release`, `debug` (ASan+UBSan), `profile`, `fuzz` (clang libFuzzer).
GPU backends layer onto any preset:
- Vulkan: `-DPF_VULKAN=ON` (needs Vulkan headers/loader + glslc).
- CUDA: `-DPF_CUDA=ON` (needs the CUDA toolkit). ggml picks sensible
  `CMAKE_CUDA_ARCHITECTURES`; for a bleeding-edge GPU whose features ptxas
  rejects under the generic arch (e.g. Blackwell sm_120 → `sm_120a`), pass
  `-DCMAKE_CUDA_ARCHITECTURES=120a`.

## Run

```sh
build/release/pf-cli --info model.gguf
echo "Contact John Doe at jdoe@example.com" | \
  build/release/pf-cli --classify model.gguf 0.5       # [cpu|cuda|vulkan]
```

## Convert

Pre-converted GGUFs are linked above. To convert an `OpenAIPrivacyFilter` HF
checkpoint yourself:

```sh
pip install -r scripts/requirements.txt   # torch + safetensors + gguf
python scripts/convert.py --model <hf-model-dir> --outfile model-f16.gguf
python scripts/convert.py --model <hf-model-dir> --outfile model-f32.gguf --outtype f32
```

[`scripts/convert.py`](scripts/convert.py) reads `config.json` +
`model.safetensors` + `tokenizer.json` and emits the GGUF directly — it does
**not** depend on llama.cpp or its converter. The nightly CI converts the model
this way and gates the result against the HF reference logits, so the converter
stays in parity (`.github/workflows/ci.yml`).

## C API

Flat C API in [`include/pf.h`](include/pf.h): an opaque `pf_ctx` handle and
caller-owned flat buffers. No exceptions cross the boundary — pointer-returning
calls report failure via `pf_last_error`, every free is NULL-safe — so it binds
cleanly from other languages (purego, ctypes, cgo).

```c
#include "pf.h"
#include <string.h>
#include <stdio.h>

// device: NULL/"cpu" | "gpu" | "cuda" | "vulkan" (optionally ":N").
// n_threads <= 0 picks a default (CPU only).
pf_ctx * ctx = pf_load("model.gguf", NULL, 0);
if (pf_last_error(ctx)) { fprintf(stderr, "%s\n", pf_last_error(ctx)); return 1; }

const char * text = "Contact John Doe at jdoe@example.com";
pf_entity * ents = NULL;
size_t n = 0;
if (pf_classify(ctx, text, strlen(text), /*threshold=*/0.5f, &ents, &n) == 0) {
    for (size_t i = 0; i < n; i++)
        // start/end are byte offsets into `text`; label is valid until pf_free
        printf("%-12s [%d,%d) %.2f  %.*s\n", ents[i].label, ents[i].start,
               ents[i].end, ents[i].score, ents[i].end - ents[i].start,
               text + ents[i].start);
}
pf_entities_free(ents, n);
pf_free(ctx);
```

- `pf_classify` → `pf_entity` spans (byte offsets into the original UTF-8 text,
  score, label); spans scoring below `threshold` are dropped. `*out` is
  malloc'd — release with `pf_entities_free`.
- `pf_set_window(ctx, max_forward_tokens)` — tokens per forward pass (default
  4096). Longer inputs run as overlapping halo windows, exact because the halo
  covers the model's full receptive field; must be `> 2048` to window.
- Lower-level, for tests / FFI: `pf_tokenize` (token ids + `2n` start/end byte
  offsets) and `pf_logits` (`n * n_labels` per-token classifier logits). Free
  those flat buffers with `pf_buf_free`.
- `pf_abi_version()` / `PF_ABI_VERSION` for ABI compatibility checks.

## Verify

```sh
ctest --preset debug -LE model            # fast suite, sanitizers, no assets
# reference fixtures + GGUF (one-time, pinned env: scripts/requirements.txt):
python scripts/hf_dump.py --model <hf-model-dir> --out tests/fixtures/hf
python scripts/convert.py --model <hf-model-dir> --outfile ggufs/pf-rope2-f16.gguf
python scripts/convert.py --model <hf-model-dir> --outfile ggufs/pf-f32.gguf --outtype f32
PF_GGUF_DIR=ggufs ctest --preset release                     # full parity (f16 + tight f32)
PF_DEVICE=vulkan PF_GGUF_DIR=... ctest --preset release -L model   # on GPU
```

Measured parity (all four fixture cases, incl. a 3k-token document):
- f32 GGUF vs HF reference taps: all 91 layer taps OK, expert routing exact,
  final logits cosine 1.000000 (`scripts/compare_taps.py`).
- f16 GGUF end-to-end: argmax 100% (reference-tie carve-out), cosine >= 0.999.
- Vulkan runs at ggml's fp16 matmul precision: cosine >= 0.9985, identical
  span sets; gates widen accordingly (`PF_DEVICE`).
- Tokenizer vs HF tokenizers: 4 fixture cases + 38-text torture corpus +
  100k random differential strings — zero id/offset mismatches
  (`scripts/hf_tok_diff.py`).

## Fuzz

```sh
cmake --preset fuzz && cmake --build --preset fuzz -j
PF_GGUF=model.gguf ./build/fuzz/fuzz_tokenizer corpus_tok/
./build/fuzz/fuzz_gguf corpus_gguf/
```
