# live_gpt

**v1.0.0** — Welvet Parallel **cameral 4–15**. Lucy PDF lands at
`results/live_gpt-v1.0.0-lucy-report.pdf`. **v0.5.0** was the 1–3 named-arch freeze.

Cameral width is a tide permute axis (`-cams 4-15`, or `8`, or `1-3`). Other hosts
can pass any range; ocean/dash label `cameral×N` from the cell. `live_mnist` is
unchanged.

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

Train is **softmax CE** (`TrainStackCE`: softmax(logits) − one-hot, mean over batch), then the same Welvet credit walk as every other mode. MSE on a 65-way one-hot stays uniform and never leaves chance.

## Network

**cameral×1** (`-cams 1`)
```
Embedding (vocab × 32)
  → residual causal MHA (32 dim, 4 heads, RoPE)
  → View [B, 32×32]
  → Dense → vocab
```

**cameral×N** (N≥2) — same residual MHA stem, then Dense → Parallel(N×Dense, add) → Dense → vocab.

Backend: **SIMD**. Default train set: **4096** non-overlapping 32-char windows from the 80% split (`-train-n 0` for all).

## Matrix

Default **`-mode sprint`**: all dtypes × **FormatNone** × all Welvet train modes × **cams 4–15**.
**No k-quants.** (`-mode full` is an alias of sprint.)

| Flag | Default | Meaning |
|------|---------|---------|
| `-mode` | `sprint` | `sprint` \| `screen` \| `smoke` |
| `-train-n` | `4096` | windows per cell (`0` = all ~80%) |
| `-cams` | `4-15` | Welvet Parallel branch counts (`4-15`, `8`, `1-3`, `cameral×12`) |
| `-micro` | `8` | batch (must stay 8; View is `[8, 1024]`) |
| `-data` | `data` | tinyshakespeare cache |
| `-ckpt` | `checkpoint` | progress + inflight weights |

`screen` = Lucy 6 × cameral×4 × FormatNone dtypes. `smoke` = 3 dtypes × all modes × cams 4 and 15.

Cell IDs changed from v0.5.0 (`|single|` / `|bicameral|`) to `|cameral×N|`. Use `-fresh` for a new epoch-1 sweep.

## Run

```bash
cd live_gpt
go run . -addr 0.0.0.0:8155
# open the dash, press Start
go run . -mode smoke -autostart   # tiny probe
go run . -mode screen
go run . -cams 8                  # one width
go run . -fresh                   # new 4–15 IDs; do not resume v0.5.0 cells
```

First run downloads Karpathy’s tinyshakespeare into `data/`. Re-run resumes; `-fresh` wipes.

After a finished epoch the Lucy PDF is written under `results/`. From an
existing checkpoint (no train):

```bash
go run . -pdf
# → results/live_gpt-v1.0.0-lucy-report.pdf
```

## Stack

- **tide** — serve+train, Lucy metrics, HTML pulse  
- **Welvet** — Embedding + causal MHA + Dense / Parallel  
- Data lineage: nanoGPT `shakespeare_char` (not OpenWebText, not GPT-2 124M)
