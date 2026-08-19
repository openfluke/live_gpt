package main

import (
	"testing"

	"github.com/openfluke/tide/permute"
	"github.com/openfluke/welvet/core"
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
