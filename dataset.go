package main

import (
	"math/rand/v2"
	"sync"

	"github.com/openfluke/tide/runner"
	"github.com/openfluke/welvet/core"
)

// Window is one nanoGPT-style block: SeqLen context tokens → next-char class.
type Window struct {
	Ctx  []int // length SeqLen
	Next int
}

type split struct {
	Train []Window
	Val   []Window
	Vocab int
}

// tideDS: serve = random val windows; train = one shuffled pass over train windows.
type tideDS struct {
	mu     sync.Mutex
	train  []Window
	val    []Window
	vocab  int
	seqLen int
	batch  int
	seed   uint64
	rng    *rand.Rand
	offset int
	order  []int
}

func windowsFromIDs(ids []int, seqLen int) []Window {
	if seqLen < 1 {
		seqLen = 32
	}
	n := len(ids) - seqLen
	if n < 1 {
		return nil
	}
	out := make([]Window, 0, n/seqLen+1)
	for i := 0; i+seqLen < len(ids); i += seqLen {
		ctx := make([]int, seqLen)
		copy(ctx, ids[i:i+seqLen])
		out = append(out, Window{Ctx: ctx, Next: ids[i+seqLen]})
	}
	return out
}

func makeSplit(c *Corpus, seqLen, trainN int, seed uint64) *split {
	ids := c.IDs
	cut := len(ids) * 4 / 5
	if cut < seqLen+1 {
		cut = seqLen + 1
	}
	if cut > len(ids) {
		cut = len(ids)
	}
	trainW := windowsFromIDs(ids[:cut], seqLen)
	valW := windowsFromIDs(ids[cut:], seqLen)
	if len(valW) == 0 && len(trainW) > 1 {
		valW = trainW[len(trainW)/5:]
		trainW = trainW[:len(trainW)*4/5]
	}
	if trainN > 0 && trainN < len(trainW) {
		shuf := rand.New(rand.NewPCG(seed, seed^0x5348414B))
		order := make([]int, len(trainW))
		for i := range order {
			order[i] = i
		}
		for i := len(order) - 1; i > 0; i-- {
			j := shuf.IntN(i + 1)
			order[i], order[j] = order[j], order[i]
		}
		take := make([]Window, trainN)
		for i := 0; i < trainN; i++ {
			take[i] = trainW[order[i]]
		}
		trainW = take
	}
	return &split{Train: trainW, Val: valW, Vocab: c.Vocab()}
}

func newTideDS(sp *split, seqLen, batch int, seed uint64) *tideDS {
	if batch < 1 {
		batch = 8
	}
	d := &tideDS{
		train:  sp.Train,
		val:    sp.Val,
		vocab:  sp.Vocab,
		seqLen: seqLen,
		batch:  batch,
		seed:   seed,
		rng:    rand.New(rand.NewPCG(seed, seed^0xC0FFEE)),
	}
	d.ResetEpoch(0)
	return d
}

func (d *tideDS) NextServe(phase string) runner.Sample {
	_ = phase
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.pack(d.val, true)
}

func (d *tideDS) TrainLen() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return len(d.train)
}

func (d *tideDS) EpochOffset() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.offset
}

func (d *tideDS) ResetEpoch(offset int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	n := len(d.train)
	d.order = make([]int, n)
	for i := range d.order {
		d.order[i] = i
	}
	shuf := rand.New(rand.NewPCG(d.seed, d.seed^0xE90C4))
	for i := n - 1; i > 0; i-- {
		j := shuf.IntN(i + 1)
		d.order[i], d.order[j] = d.order[j], d.order[i]
	}
	if offset < 0 {
		offset = 0
	}
	if offset > n {
		offset = n
	}
	d.offset = offset
}

func (d *tideDS) NextTrain() (runner.Sample, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	n := len(d.train)
	if d.offset+d.batch > n {
		return runner.Sample{}, false
	}
	batch := make([]Window, d.batch)
	for i := 0; i < d.batch; i++ {
		batch[i] = d.train[d.order[d.offset]]
		d.offset++
	}
	return d.packWindows(batch), true
}

func (d *tideDS) pack(pool []Window, random bool) runner.Sample {
	n := d.batch
	if n > len(pool) {
		n = len(pool)
	}
	batch := make([]Window, n)
	for i := 0; i < n; i++ {
		if random {
			batch[i] = pool[d.rng.IntN(len(pool))]
		} else {
			batch[i] = pool[i]
		}
	}
	return d.packWindows(batch)
}

func (d *tideDS) packWindows(batch []Window) runner.Sample {
	b := len(batch)
	t := d.seqLen
	x := core.NewTensor[float32](b, t)
	target := core.NewTensor[float32](b, d.vocab)
	labels := make([]int, b)
	for i, w := range batch {
		for j := 0; j < t && j < len(w.Ctx); j++ {
			x.Data[i*t+j] = float32(w.Ctx[j])
		}
		lab := w.Next
		if lab < 0 {
			lab = 0
		}
		if lab >= d.vocab {
			lab %= d.vocab
		}
		labels[i] = lab
		target.Data[i*d.vocab+lab] = 1
	}
	return runner.Sample{X: x, Target: target, Labels: labels}
}
