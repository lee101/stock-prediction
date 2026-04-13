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
	case "bench":
		runBench()
	case "cmaes":
		runCMAES()
	default:
		fmt.Println("binancego simcli -- trading simulation toolkit")
		fmt.Println()
		fmt.Println("Commands:")
		fmt.Println("  sim     Run a single simulation on a CSV file")
		fmt.Println("  sweep   Sweep epochs/configs in parallel")
		fmt.Println("  bench   Benchmark simulator throughput")
		fmt.Println("  cmaes   Optimize sim params with CMA-ES")
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

	// Generate dummy actions for testing (buy 0.5% below close, sell 0.5% above)
	actions := make([]sim.Action, len(simBars))
	for i, b := range simBars {
		actions[i] = sim.Action{
			BuyPrice:  b.Close * 0.995,
			SellPrice: b.Close * 1.005,
			BuyAmount: 50,
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
	dataRoot := fs.String("data-root", "trainingdatahourly/crypto", "data root for OHLCV CSVs")
	symbols := fs.String("symbols", "BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD", "comma-separated symbols")
	lags := fs.String("lags", "0,1,2", "comma-separated decision lags")
	windows := fs.String("windows", "14,30,60", "comma-separated window days")
	fee := fs.Float64("fee", 0.001, "maker fee rate")
	workers := fs.Int("workers", runtime.NumCPU(), "num workers")
	output := fs.String("output", "sweep_results.json", "output JSON path")
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

func runBench() {
	fmt.Printf("Benchmarking on %d cores...\n", runtime.NumCPU())

	// Generate 30-day data
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

	// Single-threaded benchmark
	nIter := 10000
	start := time.Now()
	for i := 0; i < nIter; i++ {
		sim.Simulate(bars, actions, cfg)
	}
	elapsed := time.Since(start)
	simsPerSec := float64(nIter) / elapsed.Seconds()
	fmt.Printf("Single-thread: %.0f sims/sec (%.1f us/sim)\n",
		simsPerSec, elapsed.Seconds()/float64(nIter)*1e6)

	// Multi-threaded benchmark
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
	metric := fs.String("metric", "sortino", "target metric (sortino or total_return)")
	fs.Parse(os.Args[2:])

	if *csvPath == "" {
		log.Fatal("--data required")
	}

	bars, err := data.LoadCSV(*csvPath)
	if err != nil {
		log.Fatalf("load csv: %v", err)
	}

	simBars := data.ToSimBars(bars)
	// Dummy actions
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
