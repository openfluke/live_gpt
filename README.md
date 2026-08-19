# live_gpt

**v0.5.0** — freeze of **single / bicameral / tricameral** (1–3 heads). Lucy PDF
from this matrix is `results/live_gpt-v0.5.0-lucy-report.pdf`. **v1.0.0** is
reserved for 4+ camerals (that sweep will change the PDF).

Live mid-stream adaptation benchmark on **Tiny Shakespeare** (the same
character stream [nanoGPT](https://github.com/karpathy/nanoGPT) uses for
`shakespeare_char`). One host of the [`tide`](../tide) serve+train engine.

Stem is **causal MHA** (GPT decoder self-attn), not the MNIST CNN.
`live_mnist` is unchanged (`Config.Build` stays nil → `chain.Model`).

---

## Task

Next-character classification:

```
context [B, 32] token ids  →  Embedding → causal MHA → flatten → Dense → vocab logits
label  = char after the window
```

Phase **A → B → A2** uses tide’s usual remap: on B, `label = (label+5) % vocab`
(Caesar on the next char). That is the MNIST `+5` shock for a language head.

SoftAcc chance is ~`100/vocab` (~1.5% on ~65 chars), not 25% 4-way XOR.

## Network

**single** (`-arches cnn`)
```
Embedding (vocab × 32)
  → MHA DecoderCausal (32 dim, 4 heads, RoPE, causal)
  → View [B, 32×32]
  → Dense → vocab
```

**bicameral / tricameral** — same MHA stem, then Dense → Parallel(n×Dense, add) → Dense → vocab.

Backend: **SIMD**. Default train set: **4096** non-overlapping 32-char windows from the 80% split (`-train-n 0` for all).

## Matrix

Default **`-mode sprint`**: all dtypes × **FormatNone** × all Welvet train modes × single/bi/tri.
**No k-quants.** (`-mode full` is an alias of sprint.)

| Flag | Default | Meaning |
|------|---------|---------|
| `-mode` | `sprint` | `sprint` \| `screen` \| `smoke` |
| `-train-n` | `4096` | windows per cell (`0` = all ~80%) |
| `-arches` | all | `cnn` (single), `bicameral`, `tricameral` |
| `-micro` | `8` | batch (must stay 8; View is `[8, 1024]`) |
| `-data` | `data` | tinyshakespeare cache |
| `-ckpt` | `checkpoint` | progress + inflight weights |

`screen` = Lucy 6 × single × all dtypes. `smoke` = 3 dtypes × all modes × 3 arches.

## Run

```bash
cd live_gpt
go run . -addr 0.0.0.0:8155
# open the dash, press Start
go run . -mode smoke -autostart   # tiny probe
go run . -mode screen -arches cnn
```

First run downloads Karpathy’s tinyshakespeare into `data/`. Re-run resumes; `-fresh` wipes.

After a finished epoch the Lucy PDF is written under `results/`. From an
existing checkpoint (no train):

```bash
go run . -pdf
# → results/live_gpt-v0.5.0-lucy-report.pdf
```

## Stack

- **tide** — serve+train, Lucy metrics, HTML pulse  
- **Welvet** — Embedding + causal MHA + Dense / Parallel  
- Data lineage: nanoGPT `shakespeare_char` (not OpenWebText, not GPT-2 124M)
