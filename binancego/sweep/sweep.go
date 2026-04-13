package sweep

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"binancego/data"
	"binancego/sim"
)

// SweepConfig configures an epoch sweep.
type SweepConfig struct {
	CheckpointDir string
	Symbols       []string
	Lags          []int
	WindowDays    []int
	FeeRate       float64
	MaxHoldBars   int
	FillBufferPct float64
	IntensityScale float64
	DataRoot      string // e.g. "trainingdatahourly/crypto"
	NumWorkers    int
}

func DefaultSweepConfig() SweepConfig {
	return SweepConfig{
		Symbols:        []string{"BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AAVEUSD", "LINKUSD"},
		Lags:           []int{0, 1, 2},
		WindowDays:     []int{14, 30, 60},
		FeeRate:        0.001,
		MaxHoldBars:    24,
		FillBufferPct:  0.0005,
		IntensityScale: 1.0,
		DataRoot:       "trainingdatahourly/crypto",
	}
}

// EpochResult holds results for one epoch across symbols/lags/windows.
type EpochResult struct {
	Epoch       string                `json:"epoch"`
	Lag         int                   `json:"lag"`
	WindowDays  int                   `json:"window_days"`
	MeanReturn  float64               `json:"mean_return_pct"`
	MeanSortino float64               `json:"mean_sortino"`
	PosSym      int                   `json:"positive_symbols"`
	TotalTrades int                   `json:"total_trades"`
	WorstDD     float64               `json:"worst_dd_pct"`
	PerSymbol   map[string]SymResult  `json:"per_symbol"`
}

// SymResult holds per-symbol results.
type SymResult struct {
	RetPct  float64 `json:"ret_pct"`
	Sortino float64 `json:"sortino"`
	Trades  int     `json:"trades"`
	DDPct   float64 `json:"dd_pct"`
}

// RunActionSweep runs a sweep using pre-computed action CSV files.
// actionsDir should contain files like: epoch_005_BTCUSD_actions.csv
func RunActionSweep(cfg SweepConfig, actionsDir string) ([]EpochResult, error) {
	pool := NewWorkerPool(cfg.NumWorkers)

	// Load bars for each symbol
	barsBySymbol := make(map[string][]sim.Bar)
	for _, sym := range cfg.Symbols {
		csvPath := filepath.Join(cfg.DataRoot, sym+".csv")
		bars, err := data.LoadCSV(csvPath)
		if err != nil {
			fmt.Printf("  %s: SKIP (%v)\n", sym, err)
			continue
		}
		barsBySymbol[sym] = data.ToSimBars(bars)
	}

	// Find all action files
	actionFiles, _ := filepath.Glob(filepath.Join(actionsDir, "*.csv"))
	if len(actionFiles) == 0 {
		return nil, fmt.Errorf("no action files in %s", actionsDir)
	}

	// Parse action file names to get epochs
	epochs := make(map[string]bool)
	for _, f := range actionFiles {
		base := filepath.Base(f)
		parts := strings.Split(strings.TrimSuffix(base, ".csv"), "_")
		if len(parts) >= 2 {
			ep := parts[0] + "_" + parts[1] // e.g. "epoch_005"
			epochs[ep] = true
		}
	}

	var allJobs []SimJob
	type jobMeta struct {
		Epoch  string
		Symbol string
		Lag    int
		Window int
	}
	var metas []jobMeta

	jobID := 0
	for ep := range epochs {
		for _, sym := range cfg.Symbols {
			bars, ok := barsBySymbol[sym]
			if !ok {
				continue
			}
			// Load actions for this epoch+symbol
			actionPath := filepath.Join(actionsDir, ep+"_"+sym+"_actions.csv")
			actBars, err := data.LoadCSV(actionPath)
			if err != nil {
				continue
			}
			actions := make([]sim.Action, len(actBars))
			for i, b := range actBars {
				actions[i] = sim.Action{
					BuyPrice:  b.Open,  // placeholder: real actions would have proper columns
					SellPrice: b.High,
					BuyAmount: b.Low,
					SellAmount: b.Close,
				}
			}

			for _, lag := range cfg.Lags {
				for _, windowD := range cfg.WindowDays {
					windowH := windowD * 24
					bSlice := bars
					aSlice := actions
					if len(bSlice) > windowH {
						bSlice = bSlice[len(bSlice)-windowH:]
					}
					if len(aSlice) > windowH {
						aSlice = aSlice[len(aSlice)-windowH:]
					}
					bSlice, aSlice = sim.ApplyDecisionLag(bSlice, aSlice, lag)
					if bSlice == nil {
						continue
					}

					simCfg := sim.SimConfig{
						MaxLeverage:    2.0,
						MakerFee:       cfg.FeeRate,
						InitialCash:    10000,
						FillBufferPct:  cfg.FillBufferPct,
						MaxHoldBars:    cfg.MaxHoldBars,
						IntensityScale: cfg.IntensityScale,
					}

					allJobs = append(allJobs, SimJob{
						ID:      jobID,
						Bars:    bSlice,
						Actions: aSlice,
						Config:  simCfg,
						Label:   fmt.Sprintf("%s/%s/lag=%d/%dd", ep, sym, lag, windowD),
					})
					metas = append(metas, jobMeta{Epoch: ep, Symbol: sym, Lag: lag, Window: windowD})
					jobID++
				}
			}
		}
	}

	if len(allJobs) == 0 {
		return nil, fmt.Errorf("no valid jobs generated")
	}

	fmt.Printf("Running %d simulation jobs on %d workers...\n", len(allJobs), pool.NumWorkers)
	results := pool.Run(allJobs)

	// Aggregate by (epoch, lag, window)
	type aggKey struct {
		Epoch  string
		Lag    int
		Window int
	}
	agg := make(map[aggKey]map[string]sim.SimResult)
	for i, r := range results {
		m := metas[i]
		key := aggKey{m.Epoch, m.Lag, m.Window}
		if agg[key] == nil {
			agg[key] = make(map[string]sim.SimResult)
		}
		agg[key][m.Symbol] = r.Result
	}

	var epochResults []EpochResult
	for key, symResults := range agg {
		var rets, sorts []float64
		totalTrades := 0
		worstDD := 0.0
		perSym := make(map[string]SymResult)
		posSym := 0

		for sym, r := range symResults {
			rets = append(rets, r.TotalReturn)
			sorts = append(sorts, r.Sortino)
			totalTrades += r.NumTrades
			if r.MaxDrawdown < worstDD {
				worstDD = r.MaxDrawdown
			}
			if r.TotalReturn > 0 {
				posSym++
			}
			perSym[sym] = SymResult{
				RetPct:  r.TotalReturn * 100,
				Sortino: r.Sortino,
				Trades:  r.NumTrades,
				DDPct:   r.MaxDrawdown * 100,
			}
		}

		meanRet := avg(rets)
		meanSort := avg(sorts)

		epochResults = append(epochResults, EpochResult{
			Epoch:       key.Epoch,
			Lag:         key.Lag,
			WindowDays:  key.Window,
			MeanReturn:  meanRet * 100,
			MeanSortino: meanSort,
			PosSym:      posSym,
			TotalTrades: totalTrades,
			WorstDD:     worstDD * 100,
			PerSymbol:   perSym,
		})
	}

	sort.Slice(epochResults, func(i, j int) bool {
		if epochResults[i].Epoch != epochResults[j].Epoch {
			return epochResults[i].Epoch < epochResults[j].Epoch
		}
		if epochResults[i].Lag != epochResults[j].Lag {
			return epochResults[i].Lag < epochResults[j].Lag
		}
		return epochResults[i].WindowDays < epochResults[j].WindowDays
	})

	return epochResults, nil
}

// SaveResults writes results to JSON.
func SaveResults(results []EpochResult, path string) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// PrintSummary prints a human-readable summary.
func PrintSummary(results []EpochResult) {
	for _, r := range results {
		symStr := ""
		for sym, sr := range r.PerSymbol {
			if symStr != "" {
				symStr += " | "
			}
			symStr += fmt.Sprintf("%s:%+.2f%%", sym, sr.RetPct)
		}
		fmt.Printf("  %s lag=%d %dd: ret=%+.3f%% sort=%.2f pos=%d trades=%d dd=%.1f%% [%s]\n",
			r.Epoch, r.Lag, r.WindowDays, r.MeanReturn, r.MeanSortino,
			r.PosSym, r.TotalTrades, r.WorstDD, symStr)
	}
}

func avg(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}
