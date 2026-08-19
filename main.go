// live_gpt — nanoGPT-style char LM adaptation sweep via tide + Welvet MHA.
//
// Tiny Shakespeare (same file as karpathy/nanoGPT shakespeare_char). Next-char
// classification over a 32-token context. Stem is causal MHA (not CNN).
// Default matrix: all dtypes × FormatNone × all train modes × cameral 4–15.
// No k-quants. live_mnist is untouched.
package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/openfluke/tide/checkpoint"
	"github.com/openfluke/tide/dash"
	"github.com/openfluke/tide/permute"
	"github.com/openfluke/tide/pulse"
	"github.com/openfluke/tide/report"
	"github.com/openfluke/tide/runner"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
)

// Version is the live_gpt freeze this binary reports. v1.0.0 is Welvet Parallel
// cameral 4–15 (v0.5.0 was named 1–3).
const Version = "1.0.0"

func main() {
	addr := flag.String("addr", "0.0.0.0:8155", "dashboard listen address (0.0.0.0 = all interfaces)")
	mode := flag.String("mode", "sprint", "permutation set: sprint | screen | smoke  (no k-quants)")
	cams := flag.String("cams", "", "cameral counts (Welvet Parallel branches). Default 4-15. e.g. 4-15, 8, 1-3")
	arches := flag.String("arches", "", "alias of -cams")
	trainN := flag.Int("train-n", 4096, "train windows per cell (0 = all 80% windows)")
	dataDir := flag.String("data", "data", "tinyshakespeare cache directory")
	ckptDir := flag.String("ckpt", "checkpoint", "progress + model checkpoint directory")
	batch := flag.Int("batch", 4, "permutations per dashboard batch")
	micro := flag.Int("micro", 8, "LM micro-batch (must match the View baked into the stack)")
	ckptSec := flag.Int("ckpt-sec", 60, "seconds between model/score checkpoints")
	lr := flag.Float64("lr", 0.02, "learning rate")
	fresh := flag.Bool("fresh", false, "ignore existing checkpoint and start clean at epoch 1")
	autostart := flag.Bool("autostart", false, "start training immediately (skip dashboard Start button)")
	pdfOnly := flag.Bool("pdf", false, "write Lucy PDF from checkpoint and exit (no training)")
	pdfOut := flag.String("pdf-out", "", "PDF path (default results/live_gpt-v"+Version+"-lucy-report.pdf)")
	flag.Parse()

	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Printf(" live_gpt v%s — tide × Welvet causal MHA @ SIMD\n", Version)
	fmt.Println(" data: tinyshakespeare char LM (nanoGPT shakespeare_char)")
	fmt.Println(" net:  Embedding → causal MHA → Welvet Parallel cameral head → vocab")
	fmt.Println(" arch: Welvet Parallel cameral×N  (-cams; sprint default 4–15)")
	fmt.Println(" Score = Throughput × Availability × Acc / 10_000")
	fmt.Println(" phase B: next-char +5 mod vocab (same Lucy remap as MNIST)")
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Printf(" SIMD linked: %v\n", simd.Enabled())
	fmt.Printf(" Dashboard:   %s\n", dashURLs(*addr))
	fmt.Printf(" Mode:        %s\n", *mode)
	fmt.Printf(" Checkpoint:  %s (every %ds)\n\n", *ckptDir, *ckptSec)

	pcfg, err := matrix(*mode)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if use := strings.TrimSpace(*cams); use != "" || strings.TrimSpace(*arches) != "" {
		if use == "" {
			use = strings.TrimSpace(*arches)
		}
		n, err := permute.ParseCams(use)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		if len(n) > 0 {
			pcfg.Cams = n
		}
	}
	cells := permute.Expand(pcfg)
	fmt.Printf(" Cameral:      %s\n", camLabel(pcfg.Cams))
	store := checkpoint.New(*ckptDir, *mode)
	var resume *checkpoint.Progress
	if !*fresh {
		resume, err = store.Load()
		if err != nil {
			fmt.Fprintln(os.Stderr, "checkpoint load:", err)
			os.Exit(1)
		}
	}
	if *pdfOnly {
		writePDFAndExit(resume, cells, *addr, *pdfOut)
		return
	}

	fmt.Printf(" Permutations: %d (batch size %d)  dtypes×FormatNone×modes×cams\n", len(cells), *batch)
	fmt.Println("Loading tinyshakespeare…")
	corp, err := LoadShakespeare(*dataDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	sp := makeSplit(corp, seqLen, *trainN, 0x475054)
	fmt.Printf("  vocab=%d  seq=%d  train_windows=%d  val_windows=%d\n", sp.Vocab, seqLen, len(sp.Train), len(sp.Val))
	fmt.Printf("  → each cell trains until all %d windows are seen once (flips at 1/3 and 2/3)\n\n", len(sp.Train))

	if resume != nil && resume.Inflight != nil && resume.Inflight.TrainOffset > len(sp.Train) {
		fmt.Fprintf(os.Stderr, "inflight cell %s is at example %d but this run only has %d train windows.\n",
			resume.Inflight.Cell.ID, resume.Inflight.TrainOffset, len(sp.Train))
		fmt.Fprintf(os.Stderr, "Pass -train-n 0 to finish it, or -fresh to drop inflight.\n")
		os.Exit(2)
	}
	epoch, resume := checkpoint.PrepareEpoch(resume, cells)
	if resume != nil {
		done := checkpoint.DoneSet(resume)
		fmt.Printf("  epoch %d — resumed: %d/%d cells done", epoch, len(done), len(cells))
		if resume.Inflight != nil {
			fmt.Printf(", inflight=%s @%d/%d", resume.Inflight.Cell.ID, resume.Inflight.TrainOffset, len(sp.Train))
		}
		fmt.Println()
		printBests("  checkpoint bests (raw)", resume.Best)
		printMobile("  checkpoint bests (mobile = metric/MiB)", resume.BestMobile)
	} else {
		fmt.Printf("  epoch %d — fresh sweep\n", epoch)
	}

	tr := pulse.New()
	srv := &dash.Server{
		Tracker:  tr,
		Cells:    cells,
		Addr:     *addr,
		Epoch:    epoch,
		Task:     "GPT-char",
		Subtitle: fmt.Sprintf("live_gpt v%s · %s · tinyshakespeare · %d windows · seq %d · causal MHA · A→B→A2 · SIMD", Version, camLabel(pcfg.Cams), len(sp.Train), seqLen),
	}
	go func() {
		if err := srv.ListenAndServe(); err != nil {
			fmt.Fprintln(os.Stderr, "dash:", err)
		}
	}()

	g := defaultGeo(sp.Vocab, *micro)
	cfg := runner.DefaultConfig(cells)
	cfg.BatchSize = *batch
	cfg.Epoch = epoch
	cfg.CheckpointEvery = time.Duration(*ckptSec) * time.Second
	cfg.LR = *lr
	cfg.Store = store
	cfg.Resume = resume
	cfg.Build = func(cell permute.Cell) (runner.Net, error) {
		return buildNet(cell, g)
	}

	ds := newTideDS(sp, seqLen, *micro, 0x475054)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	doneN := len(checkpoint.DoneSet(resume))
	runner.Hydrate(tr, cfg, fmt.Sprintf(
		"paused — epoch %d — %d/%d done — press Start on dashboard",
		epoch, doneN, len(cells)))

	if *autostart {
		srv.SignalStart()
		fmt.Printf("Autostart — running epoch %d.\n", epoch)
	} else {
		fmt.Printf("Dashboard ready (epoch %d) — open it and press Start/Resume.\n", epoch)
		if err := srv.AwaitStart(ctx); err != nil {
			fmt.Printf("\nStopped before start — checkpoint unchanged under %s.\n", *ckptDir)
			return
		}
		fmt.Printf("Start pressed — running epoch %d.\n", epoch)
	}

	if err := runner.Run(ctx, cfg, ds, tr); err != nil && ctx.Err() == nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	live := tr.Snapshot()
	printBests("\n── Best raw (3 metrics + Score) ──", live.Best)
	printMobile("\n── Best mobile (metric / MiB) ──", live.BestMobile)
	fmt.Println("\n── Leaderboard raw (Lucy Score) ──")
	for i, r := range live.Leaderboard {
		if i >= 15 {
			break
		}
		s := r.Snapshot
		fmt.Printf("%2d  [e%d] %-48s  score=%7.3f  soft=%5.1f  hard=%5.1f  thru=%7.1f  avail=%5.1f  adapt=%5.1f  %6.1fKiB  %s\n",
			i+1, r.Epoch, r.Cell.ID, s.Score, s.SoftAcc, s.AvgAccuracy, s.Throughput, s.Availability,
			s.AdaptPct, float64(s.WeightBytes)/1024, r.Status)
	}
	if ctx.Err() != nil {
		fmt.Printf("\nStopped — progress saved under %s (re-run to resume mid-epoch).\n", *ckptDir)
		return
	}
	if path, err := writeLivePDF(srv, *pdfOut); err != nil {
		fmt.Fprintln(os.Stderr, "pdf:", err)
	} else {
		fmt.Printf("\nWrote %s (live_gpt v%s)\n", path, Version)
	}
	fmt.Printf("\nEpoch %d complete. Re-run `go run .` for epoch %d (weights continue).\n", epoch, epoch+1)
	fmt.Println("Dashboard still serving — Ctrl+C to exit.")
	<-ctx.Done()
}

func camLabel(ns []int) string {
	if len(ns) == 0 {
		return "cameral×N"
	}
	if len(ns) == 1 {
		return fmt.Sprintf("cameral×%d", ns[0])
	}
	lo, hi := ns[0], ns[0]
	for _, n := range ns {
		if n < lo {
			lo = n
		}
		if n > hi {
			hi = n
		}
	}
	if hi-lo+1 == len(ns) {
		return fmt.Sprintf("cameral %d–%d", lo, hi)
	}
	parts := make([]string, len(ns))
	for i, n := range ns {
		parts[i] = fmt.Sprintf("%d", n)
	}
	return "cams " + strings.Join(parts, ",")
}

func camLabelFromCells(cells []permute.Cell) string {
	seen := map[int]bool{}
	var ns []int
	for _, c := range cells {
		n := c.Cams
		if n < 1 {
			n = permute.CamsOf(c.Arch)
		}
		if n < 1 || seen[n] {
			continue
		}
		seen[n] = true
		ns = append(ns, n)
	}
	return camLabel(ns)
}

func matrix(mode string) (permute.Config, error) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "sprint", "full":
		s := permute.Sprint()
		s.Cams = permute.CamsRange(4, 15)
		return s, nil
	case "screen":
		s := permute.Sprint()
		s.Modes = permute.LucyModes()
		s.Cams = []int{4}
		return s, nil
	case "smoke":
		return permute.Config{
			DTypes:  []core.DType{core.DTypeFloat32, core.DTypeFloat16, core.DTypeInt8},
			Formats: []quant.Format{quant.FormatNone},
			Modes:   permute.AllModes(),
			Cams:    []int{4, 15},
		}, nil
	default:
		return permute.Config{}, fmt.Errorf("unknown -mode %q (sprint|screen|smoke)", mode)
	}
}

func printBests(title string, b pulse.Best) {
	fmt.Println(title)
	printBestLine("score", b.Score)
	printBestLine("throughput", b.Throughput)
	printBestLine("availability", b.Availability)
	printBestLine("accuracy", b.Accuracy)
}

func printBestLine(name string, r *pulse.Result) {
	if r == nil {
		fmt.Printf("  %-12s  —\n", name)
		return
	}
	s := r.Snapshot
	fmt.Printf("  %-12s  %s  score=%.3f soft=%.1f hard=%.1f thru=%.1f avail=%.1f adapt=%.1f  (%.1f KiB)\n",
		name, r.Cell.ID, s.Score, s.SoftAcc, s.AvgAccuracy, s.Throughput, s.Availability, s.AdaptPct, float64(s.WeightBytes)/1024)
}

func printMobile(title string, b pulse.BestMobile) {
	fmt.Println(title)
	printMobileLine("score", b.Score, func(s pulse.Result) float64 { return s.Snapshot.MobileScore })
	printMobileLine("throughput", b.Throughput, func(s pulse.Result) float64 { return s.Snapshot.MobileThroughput })
	printMobileLine("availability", b.Availability, func(s pulse.Result) float64 { return s.Snapshot.MobileAvailability })
	printMobileLine("accuracy", b.Accuracy, func(s pulse.Result) float64 { return s.Snapshot.MobileAccuracy })
}

func printMobileLine(name string, r *pulse.Result, eff func(pulse.Result) float64) {
	if r == nil {
		fmt.Printf("  %-12s  —\n", name)
		return
	}
	s := r.Snapshot
	fmt.Printf("  %-12s  %s  eff=%.3f/MiB  raw score=%.3f acc=%.1f thru=%.1f  (%.1f KiB)\n",
		name, r.Cell.ID, eff(*r), s.Score, s.AvgAccuracy, s.Throughput, float64(s.WeightBytes)/1024)
}

func dashURLs(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		if strings.HasPrefix(addr, ":") {
			host, port = "", strings.TrimPrefix(addr, ":")
		} else {
			return "http://" + addr
		}
	}
	if port == "" {
		port = "8155"
	}
	all := host == "" || host == "0.0.0.0" || host == "::"
	if !all {
		return "http://" + net.JoinHostPort(host, port)
	}
	lines := []string{
		fmt.Sprintf("http://127.0.0.1:%s  (local)", port),
		fmt.Sprintf("listening on 0.0.0.0:%s — remote: http://<this-host-ip>:%s", port, port),
	}
	if ip := firstLANIPv4(); ip != "" {
		lines = append(lines, fmt.Sprintf("http://%s:%s  (LAN)", ip, port))
	}
	return strings.Join(lines, "\n              ")
}

func firstLANIPv4() string {
	ifaces, err := net.Interfaces()
	if err != nil {
		return ""
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			var ip net.IP
			switch v := a.(type) {
			case *net.IPNet:
				ip = v.IP
			case *net.IPAddr:
				ip = v.IP
			}
			if ip == nil || ip.IsLoopback() {
				continue
			}
			ip = ip.To4()
			if ip == nil {
				continue
			}
			return ip.String()
		}
	}
	return ""
}

func defaultPDFPath() string {
	return filepath.Join("results", fmt.Sprintf("live_gpt-v%s-lucy-report.pdf", Version))
}

func writePDFAndExit(resume *checkpoint.Progress, cells []permute.Cell, addr, out string) {
	if resume == nil || len(resume.Completed) == 0 {
		fmt.Fprintln(os.Stderr, "no checkpoint results — run a sweep before -pdf")
		os.Exit(1)
	}
	tr := pulse.New()
	epoch := resume.Epoch
	if epoch < 1 {
		epoch = 1
	}
	srv := &dash.Server{
		Tracker:  tr,
		Cells:    cells,
		Addr:     addr,
		Epoch:    epoch,
		ID:       "live_gpt",
		Task:     "GPT-char",
		Subtitle: fmt.Sprintf("live_gpt v%s · %s · %d finished cells · causal MHA", Version, camLabelFromCells(cells), len(resume.Completed)),
	}
	cfg := runner.DefaultConfig(cells)
	cfg.Epoch = epoch
	cfg.Resume = resume
	runner.Hydrate(tr, cfg, fmt.Sprintf("pdf v%s — %d/%d recorded", Version, len(resume.Completed), len(cells)))
	path, err := writeLivePDF(srv, out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("Wrote %s  (%d cells, epoch %d, live_gpt v%s)\n", path, len(resume.Completed), epoch, Version)
}

func writeLivePDF(srv *dash.Server, out string) (string, error) {
	if out == "" {
		out = defaultPDFPath()
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil && filepath.Dir(out) != "." {
		return "", err
	}
	pdf, err := report.PDFTide(srv.Report())
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(out, pdf, 0o644); err != nil {
		return "", err
	}
	return out, nil
}
