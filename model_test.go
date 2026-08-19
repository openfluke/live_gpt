package main

import (
	"math/rand/v2"
	"testing"

	"github.com/openfluke/tide/permute"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/quant"
)

func TestCorpusWindows(t *testing.T) {
	c := CorpusFromString(stringsRepeat("abcdefghijklmnopqrstuvwxyz\n", 20))
	if c.Vocab() < 10 {
		t.Fatalf("vocab %d", c.Vocab())
	}
	sp := makeSplit(c, 8, 0, 1)
	if len(sp.Train) < 2 || len(sp.Val) < 2 {
		t.Fatalf("windows train=%d val=%d", len(sp.Train), len(sp.Val))
	}
	ds := newTideDS(sp, 8, 2, 1)
	s := ds.NextServe("A")
	if s.X == nil || len(s.Labels) != 2 {
		t.Fatalf("serve labels=%d x=%v", len(s.Labels), s.X)
	}
	tr, ok := ds.NextTrain()
	if !ok || tr.X.Shape[1] != 8 {
		t.Fatalf("train ok=%v shape=%v", ok, tr.X.Shape)
	}
}

func TestBuildAndStep(t *testing.T) {
	c := CorpusFromString(stringsRepeat("To be or not to be, that is the question. ", 40))
	sp := makeSplit(c, seqLen, 32, 2)
	if len(sp.Train) < 16 {
		t.Fatalf("need windows, got %d vocab=%d ids=%d", len(sp.Train), sp.Vocab, len(c.IDs))
	}
	cell := permuteCell()
	g := defaultGeo(sp.Vocab, 8)
	net, err := buildNet(cell, g)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	ds := newTideDS(sp, seqLen, 8, 2)
	s := ds.NextServe("A")
	preds, soft, err := net.ServeEval(s.X, s.Target)
	if err != nil {
		t.Fatalf("serve: %v", err)
	}
	if len(preds) != 8 {
		t.Fatalf("preds %d", len(preds))
	}
	tr, ok := ds.NextTrain()
	if !ok {
		t.Fatal("no train batch")
	}
	if _, err := net.TrainStep(tr.X, tr.Target, 0.02, cell.Mode); err != nil {
		t.Fatalf("train: %v", err)
	}
	t.Logf("soft=%.2f bytes=%d", soft, net.WeightBytes())
}

func TestTrainStackCEFitsOneHot(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	h, err := dense.NewConfigured(8, 4, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, randN(8*4, rng))
	if err != nil {
		t.Fatal(err)
	}
	st, err := parallel.NewStack(h)
	if err != nil {
		t.Fatal(err)
	}
	x := core.NewTensor[float32](4, 8)
	for i := range x.Data {
		x.Data[i] = 0.2
	}
	target := core.NewTensor[float32](4, 4)
	for b := 0; b < 4; b++ {
		target.Data[b*4+1] = 1
	}
	for i := 0; i < 40; i++ {
		if _, err := parallel.TrainStackCE(st, x, target, parallel.ModeNormalBP, 0.2); err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
	}
	_, out, err := parallel.ForwardStack(st, x)
	if err != nil {
		t.Fatal(err)
	}
	okN := 0
	for b := 0; b < 4; b++ {
		best, bv := 0, out.Data[b*4]
		for c := 1; c < 4; c++ {
			if out.Data[b*4+c] > bv {
				bv, best = out.Data[b*4+c], c
			}
		}
		if best == 1 {
			okN++
		}
	}
	if okN < 4 {
		t.Fatalf("linear CE head should pick class 1, got %d/4 logits=%v", okN, out.Data)
	}
}

func TestLearnsNextChar(t *testing.T) {
	c := CorpusFromString(stringsRepeat("To be or not to be, that is the question. ", 80))
	sp := makeSplit(c, seqLen, 64, 3)
	cell := permuteCell()
	cell.Cams = 4
	cell.Arch = permute.ArchForCams(4)
	cell.ID = cell.String()
	g := defaultGeo(sp.Vocab, 8)
	net, err := buildNet(cell, g)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	ds := newTideDS(sp, seqLen, 8, 3)
	chance := 100.0 / float64(sp.Vocab)
	var last, first float64
	for i := 0; i < 80; i++ {
		tr, ok := ds.NextTrain()
		if !ok {
			ds.ResetEpoch(0)
			tr, ok = ds.NextTrain()
			if !ok {
				t.Fatal("no train batch")
			}
		}
		last, err = net.TrainStep(tr.X, tr.Target, 0.05, cell.Mode)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		if i == 0 {
			first = last
		}
	}
	s := ds.NextServe("A")
	preds, soft, err := net.ServeEval(s.X, s.Target)
	if err != nil {
		t.Fatalf("eval: %v", err)
	}
	okN := 0
	for i, p := range preds {
		if i < len(s.Labels) && p == s.Labels[i] {
			okN++
		}
	}
	acc := 100 * float64(okN) / float64(len(preds))
	t.Logf("chance=%.1f soft=%.1f acc=%.1f loss0=%.3f loss80=%.3f vocab=%d", chance, soft, acc, first, last, sp.Vocab)
	if soft < chance*2 && acc < chance*3 {
		t.Fatalf("CE should leave chance: soft=%.1f acc=%.1f chance=%.1f", soft, acc, chance)
	}
}

func TestBuildCameral15(t *testing.T) {
	c := CorpusFromString(stringsRepeat("To be or not to be, that is the question. ", 40))
	sp := makeSplit(c, seqLen, 32, 2)
	cell := permuteCell()
	cell.Cams = 15
	cell.Arch = permute.ArchForCams(15)
	cell.ID = cell.String()
	g := defaultGeo(sp.Vocab, 8)
	net, err := buildNet(cell, g)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	ds := newTideDS(sp, seqLen, 8, 2)
	s := ds.NextServe("A")
	if _, _, err := net.ServeEval(s.X, s.Target); err != nil {
		t.Fatalf("serve: %v", err)
	}
	tr, ok := ds.NextTrain()
	if !ok {
		t.Fatal("no train batch")
	}
	if _, err := net.TrainStep(tr.X, tr.Target, 0.02, cell.Mode); err != nil {
		t.Fatalf("train: %v", err)
	}
}

func stringsRepeat(s string, n int) string {
	b := make([]byte, 0, len(s)*n)
	for i := 0; i < n; i++ {
		b = append(b, s...)
	}
	return string(b)
}

func permuteCell() permute.Cell {
	c := permute.Cell{
		DType:   core.DTypeFloat32,
		Format:  quant.FormatNone,
		Mode:    permute.ModeSGD,
		Arch:    permute.ArchCNN,
		Cams:    1,
		Backend: core.BackendSIMD,
		UseSIMD: true,
	}
	c.ID = c.String()
	return c
}
