package main

import (
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"

	"github.com/openfluke/tide/metrics"
	"github.com/openfluke/tide/permute"
	"github.com/openfluke/tide/runner"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

const (
	seqLen = 32
	dModel = 32
	heads  = 4
)

// geo is the MHA / head layout for one run (vocab from the corpus).
type geo struct {
	Vocab  int
	SeqLen int
	DModel int
	Heads  int
	Batch  int
}

func defaultGeo(vocab, batch int) geo {
	if vocab < 2 {
		vocab = 2
	}
	if batch < 1 {
		batch = 8
	}
	return geo{Vocab: vocab, SeqLen: seqLen, DModel: dModel, Heads: heads, Batch: batch}
}

// StackNet is a tiny Welvet stack (embedding → causal MHA → cameral LM head).
type StackNet struct {
	Stack *parallel.Stack
}

func (n *StackNet) TrainStep(x, target *core.Tensor[float32], lr float64, mode permute.TrainMode) (loss float64, err error) {
	if n == nil || n.Stack == nil {
		return 0, fmt.Errorf("live_gpt: nil stack")
	}
	wv, err := mode.Welvet()
	if err != nil {
		return 0, err
	}
	ticks := 1
	if mode.IsStepSched() {
		ticks = 3
	}
	for i := 0; i < ticks-1; i++ {
		if _, _, err = parallel.ForwardStack(n.Stack, x); err != nil {
			return 0, err
		}
	}
	return parallel.TrainStackMSE(n.Stack, x, target, wv, lr)
}

func (n *StackNet) ServeEval(x, target *core.Tensor[float32]) (preds []int, softAcc float64, err error) {
	if n == nil || n.Stack == nil {
		return nil, 0, fmt.Errorf("live_gpt: nil stack")
	}
	_, out, err := parallel.ForwardStack(n.Stack, x)
	if err != nil {
		return nil, 0, err
	}
	if out == nil || len(out.Shape) < 2 {
		return nil, 0, fmt.Errorf("live_gpt: bad logits shape %v", out.Shape)
	}
	batch := out.Shape[0]
	classes := out.Shape[1]
	preds = make([]int, batch)
	sumSoft := 0.0
	probs := make([]float32, classes)
	for b := 0; b < batch; b++ {
		off := b * classes
		best := 0
		bv := out.Data[off]
		for c := 1; c < classes; c++ {
			v := out.Data[off+c]
			if v > bv {
				bv, best = v, c
			}
		}
		preds[b] = best
		if target != nil && len(target.Data) >= off+classes {
			lab := 0
			for c := 1; c < classes; c++ {
				if target.Data[off+c] > target.Data[off+lab] {
					lab = c
				}
			}
			softmaxInto(out.Data[off:off+classes], probs)
			sumSoft += metrics.SoftAccProb(probs[lab], 1.0)
		}
	}
	if batch > 0 {
		softAcc = sumSoft / float64(batch)
	}
	return preds, softAcc, nil
}

func (n *StackNet) WeightBytes() int64 {
	if n == nil {
		return 0
	}
	return opBytes(n.Stack)
}

func buildNet(cell permute.Cell, g geo) (runner.Net, error) {
	h := fnv.New64a()
	_, _ = h.Write([]byte(cell.ID + "mha-gpt"))
	rng := rand.New(rand.NewPCG(h.Sum64(), 0x475054))
	stack, err := buildStack(cell, g, rng)
	if err != nil {
		return nil, err
	}
	stack.Exec.Backend = core.BackendSIMD
	stack.Exec.MultiCore = true
	stack.SyncChildExec()
	if cell.Format != quant.FormatNone {
		if err := stack.Pack(cell.Format); err != nil {
			return nil, err
		}
	} else if cell.DType != core.DTypeFloat32 {
		if err := stack.SetDType(cell.DType); err != nil {
			return nil, err
		}
	}
	return &StackNet{Stack: stack}, nil
}

func buildStack(cell permute.Cell, g geo, rng *rand.Rand) (*parallel.Stack, error) {
	emb, err := embedding.NewConfigured(embedding.Config{
		VocabSize: g.Vocab, EmbeddingDim: g.DModel, SeqLen: g.SeqLen,
	}, core.DTypeFloat32, quant.FormatNone, randN(g.Vocab*g.DModel, rng))
	if err != nil {
		return nil, err
	}
	cfg := mha.DecoderCausal(g.DModel, g.Heads, g.Heads)
	cfg.MaxSeqLen = g.SeqLen
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	q := g.DModel * cfg.QDim()
	attn, err := mha.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone,
		randN(q, rng), randN(q, rng), randN(q, rng), randN(g.DModel*g.DModel, rng))
	if err != nil {
		return nil, err
	}
	flat := g.SeqLen * g.DModel
	view, err := parallel.NewView(g.Batch, flat)
	if err != nil {
		return nil, err
	}
	tail, err := classTail(flat, g.Vocab, cell, rng)
	if err != nil {
		return nil, err
	}
	kids := append([]any{emb, attn, view}, tail...)
	return parallel.NewStack(kids...)
}

func camsOf(cell permute.Cell) int {
	n := cell.Cams
	if n < 1 {
		n = permute.CamsOf(cell.Arch)
	}
	if n < 1 {
		return 1
	}
	return n
}

func classTail(inFeat, vocab int, cell permute.Cell, rng *rand.Rand) ([]any, error) {
	if camsOf(cell) < 2 {
		h, err := dense.NewConfigured(inFeat, vocab, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, randN(inFeat*vocab, rng))
		if err != nil {
			return nil, err
		}
		return []any{h}, nil
	}
	return cameralSandwich(inFeat, vocab, camsOf(cell), rng)
}

func cameralSandwich(inFeat, outFeat, cams int, rng *rand.Rand) ([]any, error) {
	hidden := inFeat
	if hidden > 64 {
		hidden = 64
	}
	if hidden < 8 {
		hidden = 8
	}
	din, err := dense.NewConfigured(inFeat, hidden, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, randN(inFeat*hidden, rng))
	if err != nil {
		return nil, err
	}
	branches := make([]any, cams)
	for i := 0; i < cams; i++ {
		b, err := dense.NewConfigured(hidden, hidden, core.ActivationTanh, core.DTypeFloat32, quant.FormatNone, randN(hidden*hidden, rng))
		if err != nil {
			return nil, fmt.Errorf("hemi %d: %w", i, err)
		}
		branches[i] = b
	}
	para, err := parallel.NewFromBranches(parallel.Config{
		Dim: hidden, OutFeat: hidden, Branches: cams, Combine: parallel.CombineAdd,
	}, branches, nil)
	if err != nil {
		return nil, err
	}
	dout, err := dense.NewConfigured(hidden, outFeat, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, randN(hidden*outFeat, rng))
	if err != nil {
		return nil, err
	}
	return []any{din, para, dout}, nil
}

func randN(n int, rng *rand.Rand) []float32 {
	w := make([]float32, n)
	scale := float32(1 / math.Sqrt(float64(n)))
	if scale > 0.1 {
		scale = 0.1
	}
	for i := range w {
		w[i] = (rng.Float32()*2 - 1) * scale
	}
	return w
}

func softmaxInto(logits, out []float32) {
	n := len(logits)
	if n == 0 || len(out) < n {
		return
	}
	max := logits[0]
	for i := 1; i < n; i++ {
		if logits[i] > max {
			max = logits[i]
		}
	}
	var sum float64
	for i := 0; i < n; i++ {
		out[i] = float32(math.Exp(float64(logits[i] - max)))
		sum += float64(out[i])
	}
	if sum <= 0 {
		for i := 0; i < n; i++ {
			out[i] = 1 / float32(n)
		}
		return
	}
	inv := float32(1 / sum)
	for i := 0; i < n; i++ {
		out[i] *= inv
	}
}

func opBytes(op any) int64 {
	if op == nil {
		return 0
	}
	switch v := op.(type) {
	case *parallel.Stack:
		var n int64
		for _, ch := range v.Children {
			n += opBytes(ch)
		}
		return n
	case *dense.Layer:
		return storeBytes(v.Weights)
	case *mha.Layer:
		return opBytes(v.Q) + opBytes(v.K) + opBytes(v.V) + opBytes(v.O)
	case *embedding.Layer:
		return storeBytes(v.Weights)
	case *parallel.Layer:
		var n int64
		for _, ch := range v.Branches {
			n += opBytes(ch)
		}
		return n
	}
	return 0
}

func storeBytes(s *weights.Store) int64 {
	if s == nil {
		return 0
	}
	n := int64(len(s.Bias) * 8)
	if s.Packed != nil {
		n += int64(len(s.Packed.Raw))
		n += int64(len(s.Packed.Scales) * 4)
		n += int64(len(s.Packed.Mins) * 4)
		n += int64(len(s.Packed.Meta))
		return n
	}
	if len(s.Native) > 0 {
		return n + int64(len(s.Native))
	}
	bits := s.DType.Bits()
	if bits <= 0 {
		bits = 32
	}
	return n + int64((s.Rows*s.Cols*bits+7)/8)
}
