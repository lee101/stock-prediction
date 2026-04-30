// mixedcli runs the bitbankgo "mixed" multi-strategy walk-forward portfolio
// backtest on Alpaca hourly stock CSVs and reports per-100d-window stats so
// we can compare against HARD RULE #1 (27%/mo median PnL on unseen data).
//
// Usage:
//
//	mixedcli --root trainingdatahourly/stocks \
//	         --symbols AAPL,MSFT,NVDA,... \
//	         --window-bars 700 --stride-bars 175 \
//	         --fee-rate 0.001 --fill-buffer-pct 0.0005 \
//	         --max-leverage 2 --can-short \
//	         --decision-lag-bars 2 \
//	         --train-hours 168 --lookback-hours 168 \
//	         --rebalance-bars 1 \
//	         --out-json mixedcli_run.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"binancego/loader"
	"binancego/policy/mixed"
)

type windowStat struct {
	StartIdx     int     `json:"start_idx"`
	EndIdx       int     `json:"end_idx"`
	StartTime    string  `json:"start_time"`
	EndTime      string  `json:"end_time"`
	Bars         int     `json:"bars"`
	TotalReturn  float64 `json:"total_return"`
	MonthlyMed   float64 `json:"monthly_median"`
	MonthlyAnn   float64 `json:"monthly_annualized"`
	Sortino      float64 `json:"sortino"`
	MaxDrawdown  float64 `json:"max_drawdown_pct"`
	NumTrades    int     `json:"num_trades"`
	NumDecisions int     `json:"num_decisions"`
}

type sweepReport struct {
	GeneratedAt   string                 `json:"generated_at"`
	Universe      []string               `json:"universe"`
	BarsPerSymbol int                    `json:"bars_per_symbol"`
	Config        map[string]interface{} `json:"config"`
	Windows       []windowStat           `json:"windows"`
	Aggregate     map[string]float64     `json:"aggregate"`
}

func main() {
	root := flag.String("root", "/nvme0n1-disk/code/stock-prediction/trainingdatahourly/stocks", "directory of <SYM>.csv files")
	symsCSV := flag.String("symbols", "", "comma-separated symbol list")
	universeFile := flag.String("universe-file", "", "newline-separated symbol file (one symbol per line)")

	windowBars := flag.Int("window-bars", 700, "bars per backtest window (~100 trading days for hourly bars at ~7 bars/day)")
	strideBars := flag.Int("stride-bars", 175, "stride between window starts in bars (~25 trading days)")
	startBarsAhead := flag.Int("start-bars-ahead", 200, "skip the first N bars of history before the first window (warm-up budget for the walk-forward selector)")

	feeRate := flag.Float64("fee-rate", 0.001, "per-side fee (0.001 = 10bps)")
	fillBufferPct := flag.Float64("fill-buffer-pct", 0.0005, "fill buffer beyond limit price")
	maxLeverage := flag.Float64("max-leverage", 2.0, "portfolio gross leverage cap")
	canShort := flag.Bool("can-short", true, "allow short positions")
	trainHours := flag.Int("train-hours", 168, "walk-forward selector training window in bars")
	lookbackHours := flag.Int("lookback-hours", 168, "stats window for portfolio scoring")
	rebalanceBars := flag.Int("rebalance-bars", 1, "rebalance every N bars")

	outJSON := flag.String("out-json", "", "if set, write JSON summary here")

	flag.Parse()

	syms, err := readSymbols(*symsCSV, *universeFile)
	if err != nil {
		die("symbols: %v", err)
	}
	if len(syms) == 0 {
		die("no symbols supplied (use --symbols or --universe-file)")
	}
	fmt.Fprintf(os.Stderr, "[mixedcli] loading %d symbols from %s ...\n", len(syms), *root)
	loaded, err := loader.LoadAlignedStockCandles(*root, syms)
	if err != nil {
		die("LoadAlignedStockCandles: %v", err)
	}
	if len(loaded) == 0 {
		die("zero symbols loaded")
	}
	n := len(loaded[0].Bars)
	fmt.Fprintf(os.Stderr, "[mixedcli] aligned: %d symbols x %d bars\n", len(loaded), n)

	cfg := mixed.PortfolioConfig{
		LookbackHours:      *lookbackHours,
		TrainHours:         *trainHours,
		FeeRate:            *feeRate,
		FillBufferPct:      *fillBufferPct,
		MaxLeverage:        *maxLeverage,
		CanShort:           *canShort,
		RebalanceEveryBars: *rebalanceBars,
	}

	// Build windows.
	if *windowBars <= 0 || *strideBars <= 0 {
		die("window-bars and stride-bars must be positive")
	}
	if n <= *startBarsAhead+*windowBars+1 {
		die("history (%d bars) too short for window=%d + warmup=%d", n, *windowBars, *startBarsAhead)
	}

	windows := []windowStat{}
	for start := *startBarsAhead; start+*windowBars < n-1; start += *strideBars {
		end := start + *windowBars
		if end >= n-1 {
			break
		}
		fmt.Fprintf(os.Stderr, "[mixedcli] window [%d, %d] (%d bars) ...\n", start, end, end-start)
		t0 := time.Now()
		res := mixed.SimulatePortfolio(loaded, start, end, cfg)
		monthlyAnn := annualisedFromMonthly(res.MonthlyMed)
		ws := windowStat{
			StartIdx:     start,
			EndIdx:       end,
			StartTime:    formatTS(loaded[0].Bars[start].Timestamp),
			EndTime:      formatTS(loaded[0].Bars[end].Timestamp),
			Bars:         end - start,
			TotalReturn:  res.TotalReturn,
			MonthlyMed:   res.MonthlyMed,
			MonthlyAnn:   monthlyAnn,
			Sortino:      res.Sortino,
			MaxDrawdown:  res.MaxDrawdown,
			NumTrades:    res.NumTrades,
			NumDecisions: res.NumDecisions,
		}
		fmt.Fprintf(os.Stderr, "  → tot=%+.2f%% mo_med=%+.2f%% sortino=%.2f maxDD=%.2f%% trades=%d (%.1fs)\n",
			ws.TotalReturn*100, ws.MonthlyMed*100, ws.Sortino, ws.MaxDrawdown, ws.NumTrades, time.Since(t0).Seconds())
		windows = append(windows, ws)
	}

	report := sweepReport{
		GeneratedAt:   time.Now().UTC().Format(time.RFC3339),
		Universe:      symbolNames(loaded),
		BarsPerSymbol: n,
		Config: map[string]interface{}{
			"window_bars":      *windowBars,
			"stride_bars":      *strideBars,
			"start_bars_ahead": *startBarsAhead,
			"fee_rate":         *feeRate,
			"fill_buffer_pct":  *fillBufferPct,
			"max_leverage":     *maxLeverage,
			"can_short":        *canShort,
			"train_hours":      *trainHours,
			"lookback_hours":   *lookbackHours,
			"rebalance_bars":   *rebalanceBars,
		},
		Windows:   windows,
		Aggregate: aggregate(windows),
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if *outJSON != "" {
		f, err := os.Create(*outJSON)
		if err != nil {
			die("create %s: %v", *outJSON, err)
		}
		defer f.Close()
		ej := json.NewEncoder(f)
		ej.SetIndent("", "  ")
		_ = ej.Encode(report)
	}
	_ = enc.Encode(report.Aggregate)
}

func aggregate(ws []windowStat) map[string]float64 {
	if len(ws) == 0 {
		return map[string]float64{}
	}
	monthlies := make([]float64, 0, len(ws))
	totals := make([]float64, 0, len(ws))
	sortinos := make([]float64, 0, len(ws))
	dds := make([]float64, 0, len(ws))
	negCount := 0
	for _, w := range ws {
		monthlies = append(monthlies, w.MonthlyMed)
		totals = append(totals, w.TotalReturn)
		sortinos = append(sortinos, w.Sortino)
		dds = append(dds, w.MaxDrawdown)
		if w.TotalReturn < 0 {
			negCount++
		}
	}
	return map[string]float64{
		"n_windows":             float64(len(ws)),
		"monthly_med_p10":       percentile(monthlies, 0.10),
		"monthly_med_median":    percentile(monthlies, 0.50),
		"monthly_med_p90":       percentile(monthlies, 0.90),
		"total_return_median":   percentile(totals, 0.50),
		"sortino_median":        percentile(sortinos, 0.50),
		"max_drawdown_p90":      percentile(dds, 0.90),
		"negative_window_count": float64(negCount),
		"negative_window_frac":  float64(negCount) / float64(len(ws)),
	}
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	tmp := make([]float64, len(values))
	copy(tmp, values)
	sort.Float64s(tmp)
	idx := int(p * float64(len(tmp)-1))
	if idx >= len(tmp) {
		idx = len(tmp) - 1
	}
	if idx < 0 {
		idx = 0
	}
	return tmp[idx]
}

func annualisedFromMonthly(monthly float64) float64 {
	return math.Pow(1+monthly, 12) - 1
}

func readSymbols(csv, file string) ([]string, error) {
	out := []string{}
	if csv != "" {
		for _, s := range strings.Split(csv, ",") {
			s = strings.TrimSpace(s)
			if s != "" {
				out = append(out, s)
			}
		}
	}
	if file != "" {
		raw, err := os.ReadFile(file)
		if err != nil {
			return nil, err
		}
		for _, line := range strings.Split(string(raw), "\n") {
			s := strings.TrimSpace(line)
			if s != "" && !strings.HasPrefix(s, "#") {
				out = append(out, s)
			}
		}
	}
	// dedupe preserving order.
	seen := map[string]bool{}
	dedup := []string{}
	for _, s := range out {
		if !seen[s] {
			seen[s] = true
			dedup = append(dedup, s)
		}
	}
	return dedup, nil
}

func symbolNames(syms []mixed.SymbolBars) []string {
	out := make([]string, len(syms))
	for i, s := range syms {
		out[i] = s.Symbol
	}
	return out
}

func formatTS(ns int64) string {
	if ns == 0 {
		return ""
	}
	return time.Unix(0, ns).UTC().Format("2006-01-02 15:04:05Z")
}

func die(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "mixedcli: "+format+"\n", args...)
	os.Exit(1)
}
