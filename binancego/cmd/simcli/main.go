package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"time"

	"binancego/data"
	"binancego/policy"
	"binancego/sim"
	"binancego/sweep"
)

func main() {
	cmd := "help"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}

	switch cmd {
	case "sim":
		runSim()
	case "sweep":
		runSweep()
	case "onnx-sweep":
		runONNXSweep()
	case "infer":
		runInfer()
	case "bench":
		runBench()
	case "cmaes":
		runCMAES()
	default:
		fmt.Println("binancego simcli -- trading simulation toolkit")
		fmt.Println()
		fmt.Println("Commands:")
		fmt.Println("  sim         Run a single simulation on a CSV file")
		fmt.Println("  sweep       Sweep epochs/configs from pre-computed actions")
		fmt.Println("  onnx-sweep  Sweep ONNX checkpoints: inference -> sim -> leaderboard")
		fmt.Println("  infer       Run ONNX inference on a CSV and print actions")
		fmt.Println("  bench       Benchmark simulator throughput")
		fmt.Println("  cmaes       Optimize sim params with CMA-ES")
	}
}

func runSim() {
	fs := flag.NewFlagSet("sim", flag.ExitOnError)
	csvPath := fs.String("data", "", "path to OHLCV CSV")
	fee := fs.Float64("fee", 0.001, "maker fee rate")
	maxHold := fs.Int("max-hold", 24, "max hold bars (0=unlimited)")
	fillBuf := fs.Float64("fill-buffer", 0.0005, "fill buffer fraction")
	leverage := fs.Float64("leverage", 2.0, "max leverage")
	lag := fs.Int("lag", 0, "decision lag bars")
	fs.Parse(os.Args[2:])

	if *csvPath == "" {
		log.Fatal("--data required")
	}

	bars, err := data.LoadCSV(*csvPath)
	if err != nil {
		log.Fatalf("load csv: %v", err)
	}

	simBars := data.ToSimBars(bars)
	actions := make([]sim.Action, len(simBars))
	for i, b := range simBars {
		actions[i] = sim.Action{
			BuyPrice:   b.Close * 0.995,
			SellPrice:  b.Close * 1.005,
			BuyAmount:  50,
			SellAmount: 80,
		}
	}

	if *lag > 0 {
		simBars, actions = sim.ApplyDecisionLag(simBars, actions, *lag)
	}

	cfg := sim.SimConfig{
		MaxLeverage:    *leverage,
		MakerFee:       *fee,
		InitialCash:    10000,
		FillBufferPct:  *fillBuf,
		MaxHoldBars:    *maxHold,
		IntensityScale: 1.0,
	}

	result := sim.Simulate(simBars, actions, cfg)
	fmt.Printf("bars=%d trades=%d\n", len(simBars), result.NumTrades)
	fmt.Printf("return=%.4f%% sortino=%.2f dd=%.4f%%\n",
		result.TotalReturn*100, result.Sortino, result.MaxDrawdown*100)
	fmt.Printf("final_equity=%.2f margin_cost=%.2f\n",
		result.FinalEquity, result.MarginCostTotal)
}

func runSweep() {
	fs := flag.NewFlagSet("sweep", flag.ExitOnError)
	actionsDir := fs.String("actions-dir", "", "directory with pre-computed action CSVs")
	dataRoot := fs.String("data-root", "trainingdatahourly/crypto", "data root")
	symbols := fs.String("symbols", "BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD", "symbols")
	lags := fs.String("lags", "0,1,2", "decision lags")
	windows := fs.String("windows", "14,30,60", "window days")
	fee := fs.Float64("fee", 0.001, "fee rate")
	workers := fs.Int("workers", runtime.NumCPU(), "workers")
	output := fs.String("output", "sweep_results.json", "output JSON")
	fs.Parse(os.Args[2:])

	if *actionsDir == "" {
		log.Fatal("--actions-dir required")
	}

	cfg := sweep.SweepConfig{
		Symbols:    strings.Split(*symbols, ","),
		Lags:       parseInts(*lags),
		WindowDays: parseInts(*windows),
		FeeRate:    *fee,
		DataRoot:   *dataRoot,
		NumWorkers: *workers,
	}

	results, err := sweep.RunActionSweep(cfg, *actionsDir)
	if err != nil {
		log.Fatalf("sweep: %v", err)
	}

	sweep.PrintSummary(results)
	if err := sweep.SaveResults(results, *output); err != nil {
		log.Printf("save results: %v", err)
	}
	fmt.Printf("\nResults saved to %s\n", *output)
}

func runONNXSweep() {
	fs := flag.NewFlagSet("onnx-sweep", flag.ExitOnError)
	ckptDir := fs.String("checkpoints", "", "directory with .onnx or .pt files")
	dataRoot := fs.String("data-root", "trainingdatahourly/crypto", "OHLCV data root")
	forecastDir := fs.String("forecasts", "", "forecast cache dir")
	symbols := fs.String("symbols", "BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD", "symbols")
	lags := fs.String("lags", "0,1,2", "decision lags")
	windows := fs.String("windows", "14,30,60", "window days")
	fee := fs.Float64("fee", 0.001, "fee rate")
	maxHold := fs.Int("max-hold", 6, "max hold bars")
	seqLen := fs.Int("seq-len", 48, "model sequence length")
	workers := fs.Int("workers", runtime.NumCPU(), "workers")
	output := fs.String("output", "onnx_sweep_results.json", "output JSON")
	fs.Parse(os.Args[2:])

	if *ckptDir == "" {
		log.Fatal("--checkpoints required")
	}

	cfg := sweep.ONNXSweepConfig{
		CheckpointDir:  *ckptDir,
		DataRoot:       *dataRoot,
		ForecastDir:    *forecastDir,
		Symbols:        strings.Split(*symbols, ","),
		Lags:           parseInts(*lags),
		WindowDays:     parseInts(*windows),
		FeeRate:        *fee,
		MaxHoldBars:    *maxHold,
		SeqLen:         *seqLen,
		NumWorkers:     *workers,
		DecodeConfig:   policy.DefaultDecodeConfig(),
	}

	results, err := sweep.RunONNXSweep(cfg)
	if err != nil {
		log.Fatalf("onnx-sweep: %v", err)
	}

	sweep.PrintLeaderboard(results)
	if err := sweep.SaveCheckpointResults(results, *output); err != nil {
		log.Printf("save: %v", err)
	}
	fmt.Printf("\n%d results saved to %s\n", len(results), *output)
}

func runInfer() {
	fs := flag.NewFlagSet("infer", flag.ExitOnError)
	modelPath := fs.String("model", "", "path to .onnx model")
	csvPath := fs.String("data", "", "path to OHLCV CSV")
	seqLen := fs.Int("seq-len", 48, "sequence length")
	metaPath := fs.String("meta", "", "training_meta.json for normalizer")
	n := fs.Int("n", 10, "number of actions to print")
	fs.Parse(os.Args[2:])

	if *modelPath == "" || *csvPath == "" {
		log.Fatal("--model and --data required")
	}

	model, err := policy.LoadONNX(*modelPath)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	defer model.Close()

	bars, err := data.LoadCSV(*csvPath)
	if err != nil {
		log.Fatalf("load csv: %v", err)
	}

	var norm *data.FeatureNormalizer
	if *metaPath != "" {
		norm, err = data.LoadNormalizerFromMeta(*metaPath)
		if err != nil {
			log.Printf("warn: normalizer: %v", err)
		}
	}

	feats := data.ComputeFeatures(bars)
	featureNames := []string{"return_1h", "volatility_24h", "volume_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "high_low_range"}

	features := make([][]float32, len(bars))
	for i := range bars {
		row := make([]float32, len(featureNames))
		for j, name := range featureNames {
			val := feats[name][i]
			if norm != nil {
				val = norm.Normalize(name, val)
			}
			row[j] = float32(val)
		}
		features[i] = row
	}

	if len(features) > *seqLen {
		features = features[len(features)-*seqLen:]
		bars = bars[len(bars)-*seqLen:]
	}

	logits, err := model.InferSequence(features)
	if err != nil {
		log.Fatalf("inference: %v", err)
	}

	cfg := policy.DefaultDecodeConfig()
	start := len(logits) - *n
	if start < 0 {
		start = 0
	}

	fmt.Printf("%-20s %10s %10s %8s %8s  logits\n", "timestamp", "buy_price", "sell_price", "buy_amt", "sell_amt")
	for i := start; i < len(logits); i++ {
		ref := bars[i].Close
		action := policy.DecodeActions(logits[i], ref, ref*1.01, ref*0.99, cfg)
		fmt.Printf("%-20s %10.2f %10.2f %8.1f %8.1f  %v\n",
			bars[i].Timestamp.Format("2006-01-02T15:04"),
			action.BuyPrice, action.SellPrice, action.BuyAmount, action.SellAmount,
			logits[i])
	}
}

func runBench() {
	fmt.Printf("Benchmarking on %d cores...\n", runtime.NumCPU())

	nBars := 720
	bars := make([]sim.Bar, nBars)
	actions := make([]sim.Action, nBars)
	for i := range bars {
		price := 50000.0 + float64(i)*10
		bars[i] = sim.Bar{price, price + 200, price - 200, price + 50}
		actions[i] = sim.Action{price - 100, price + 100, 50, 80}
	}

	cfg := sim.DefaultSimConfig()
	cfg.FillBufferPct = 0

	nIter := 10000
	start := time.Now()
	for i := 0; i < nIter; i++ {
		sim.Simulate(bars, actions, cfg)
	}
	elapsed := time.Since(start)
	simsPerSec := float64(nIter) / elapsed.Seconds()
	fmt.Printf("Single-thread: %.0f sims/sec (%.1f us/sim)\n",
		simsPerSec, elapsed.Seconds()/float64(nIter)*1e6)

	pool := sweep.NewWorkerPool(runtime.NumCPU())
	jobs := make([]sweep.SimJob, nIter)
	for i := range jobs {
		jobs[i] = sweep.SimJob{ID: i, Bars: bars, Actions: actions, Config: cfg}
	}

	start = time.Now()
	pool.Run(jobs)
	elapsed = time.Since(start)
	simsPerSec = float64(nIter) / elapsed.Seconds()
	fmt.Printf("Multi-thread (%d workers): %.0f sims/sec (%.1f us/sim)\n",
		runtime.NumCPU(), simsPerSec, elapsed.Seconds()/float64(nIter)*1e6)
}

func runCMAES() {
	fs := flag.NewFlagSet("cmaes", flag.ExitOnError)
	csvPath := fs.String("data", "", "path to OHLCV CSV")
	popSize := fs.Int("pop", 20, "population size")
	maxIter := fs.Int("iter", 50, "max iterations")
	metric := fs.String("metric", "sortino", "target metric")
	fs.Parse(os.Args[2:])

	if *csvPath == "" {
		log.Fatal("--data required")
	}

	bars, err := data.LoadCSV(*csvPath)
	if err != nil {
		log.Fatalf("load csv: %v", err)
	}

	simBars := data.ToSimBars(bars)
	actions := make([]sim.Action, len(simBars))
	for i, b := range simBars {
		actions[i] = sim.Action{b.Close * 0.995, b.Close * 1.005, 50, 80}
	}

	baseCfg := sim.DefaultSimConfig()
	cmaCfg := sweep.CMAESConfig{
		PopSize:      *popSize,
		MaxIter:      *maxIter,
		Sigma0:       0.3,
		TargetMetric: *metric,
		NumWorkers:   runtime.NumCPU(),
		Verbose:      true,
	}

	result := sweep.OptimizeCMAES(simBars, actions, baseCfg, sweep.DefaultParamBounds, cmaCfg)

	fmt.Printf("\nBest %s: %.4f\n", *metric, result.BestMetric)
	fmt.Println("Best params:")
	out, _ := json.MarshalIndent(result.BestParams, "  ", "  ")
	fmt.Println(string(out))
	fmt.Printf("Iterations: %d, Evaluations: %d\n", result.Iterations, result.Evaluations)
}

func parseInts(s string) []int {
	parts := strings.Split(s, ",")
	var result []int
	for _, p := range parts {
		var v int
		fmt.Sscanf(strings.TrimSpace(p), "%d", &v)
		result = append(result, v)
	}
	return result
}
