package sweep

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"binancego/data"
	"binancego/policy"
	"binancego/sim"
)

type ONNXSweepConfig struct {
	CheckpointDir  string
	DataRoot       string
	ForecastDir    string
	Symbols        []string
	Lags           []int
	WindowDays     []int
	FeeRate        float64
	MaxHoldBars    int
	FillBufferPct  float64
	IntensityScale float64
	SeqLen         int
	NumWorkers     int
	DecodeConfig   policy.DecodeConfig
}

func DefaultONNXSweepConfig() ONNXSweepConfig {
	return ONNXSweepConfig{
		Symbols:        []string{"BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AAVEUSD", "LINKUSD"},
		Lags:           []int{0, 1, 2},
		WindowDays:     []int{14, 30, 60},
		FeeRate:        0.001,
		MaxHoldBars:    6,
		FillBufferPct:  0.0005,
		IntensityScale: 1.0,
		SeqLen:         48,
		NumWorkers:     runtime.NumCPU(),
		DecodeConfig:   policy.DefaultDecodeConfig(),
	}
}

type CheckpointResult struct {
	Name        string                `json:"name"`
	Epoch       string                `json:"epoch"`
	Lag         int                   `json:"lag"`
	WindowDays  int                   `json:"window_days"`
	MeanReturn  float64               `json:"mean_return_pct"`
	MeanSortino float64               `json:"mean_sortino"`
	MedianSort  float64               `json:"median_sortino"`
	PosSym      int                   `json:"positive_symbols"`
	TotalSym    int                   `json:"total_symbols"`
	TotalTrades int                   `json:"total_trades"`
	WorstDD     float64               `json:"worst_dd_pct"`
	PerSymbol   map[string]SymResult  `json:"per_symbol"`
}

// RunONNXSweep loads ONNX models from a checkpoint dir, runs inference + sim for each.
func RunONNXSweep(cfg ONNXSweepConfig) ([]CheckpointResult, error) {
	onnxFiles, err := filepath.Glob(filepath.Join(cfg.CheckpointDir, "*.onnx"))
	if err != nil {
		return nil, err
	}
	if len(onnxFiles) == 0 {
		ptFiles, _ := filepath.Glob(filepath.Join(cfg.CheckpointDir, "*.pt"))
		if len(ptFiles) > 0 {
			fmt.Printf("found %d .pt checkpoints, exporting to ONNX...\n", len(ptFiles))
			for _, pt := range ptFiles {
				onnxPath, err := policy.ExportCheckpointToONNX(pt, cfg.SeqLen)
				if err != nil {
					fmt.Printf("  export %s: %v\n", filepath.Base(pt), err)
					continue
				}
				onnxFiles = append(onnxFiles, onnxPath)
			}
		}
	}
	if len(onnxFiles) == 0 {
		return nil, fmt.Errorf("no ONNX models in %s", cfg.CheckpointDir)
	}

	barsBySymbol := make(map[string][]data.TimestampedBar)
	for _, sym := range cfg.Symbols {
		csvPath := filepath.Join(cfg.DataRoot, sym+".csv")
		bars, err := data.LoadCSV(csvPath)
		if err != nil {
			fmt.Printf("  %s: SKIP (%v)\n", sym, err)
			continue
		}
		barsBySymbol[sym] = bars
	}
	if len(barsBySymbol) == 0 {
		return nil, fmt.Errorf("no data loaded from %s", cfg.DataRoot)
	}

	var forecasts map[string]*data.ForecastCache
	if cfg.ForecastDir != "" {
		forecasts, _ = data.LoadForecastDir(cfg.ForecastDir)
	}

	// Load normalizer from checkpoint dir if available
	var norm *data.FeatureNormalizer
	metaPath := filepath.Join(cfg.CheckpointDir, "training_meta.json")
	if _, err := os.Stat(metaPath); err == nil {
		norm, _ = data.LoadNormalizerFromMeta(metaPath)
	}

	pool := NewWorkerPool(cfg.NumWorkers)
	var allResults []CheckpointResult

	for _, onnxPath := range onnxFiles {
		model, err := policy.LoadONNX(onnxPath)
		if err != nil {
			fmt.Printf("  load %s: %v\n", filepath.Base(onnxPath), err)
			continue
		}

		ckptName := strings.TrimSuffix(filepath.Base(onnxPath), ".onnx")
		fmt.Printf("evaluating %s...\n", ckptName)

		// Run inference for each symbol
		actionsBySymbol := make(map[string][]sim.Action)
		for sym, bars := range barsBySymbol {
			features := buildFeatureMatrix(bars, norm, cfg.SeqLen)
			if features == nil {
				continue
			}

			logits, err := model.InferSequence(features)
			if err != nil {
				fmt.Printf("  %s/%s: inference error: %v\n", ckptName, sym, err)
				continue
			}

			actions := make([]sim.Action, len(bars))
			for i := 0; i < len(bars); i++ {
				var logitRow []float64
				if i < len(logits) {
					logitRow = logits[i]
				} else {
					logitRow = []float64{0, 0, 0, 0}
				}

				refClose := bars[i].Close
				high, low := refClose*1.01, refClose*0.99

				if forecasts != nil {
					if fc, ok := forecasts[sym]; ok {
						ts := bars[i].Timestamp.Unix()
						if row, found := fc.GetForecast(ts); found {
							high = row.PredictedHighP50
							low = row.PredictedLowP50
						}
					}
				}

				decoded := policy.DecodeActions(logitRow, refClose, high, low, cfg.DecodeConfig)
				actions[i] = sim.Action{
					BuyPrice:   decoded.BuyPrice,
					SellPrice:  decoded.SellPrice,
					BuyAmount:  decoded.BuyAmount,
					SellAmount: decoded.SellAmount,
				}
			}
			actionsBySymbol[sym] = actions
		}

		// Build sim jobs for all (lag, window) combos
		type jobMeta struct {
			Symbol string
			Lag    int
			Window int
		}
		var jobs []SimJob
		var metas []jobMeta
		jobID := 0

		for _, sym := range cfg.Symbols {
			bars, ok := barsBySymbol[sym]
			if !ok {
				continue
			}
			actions, ok := actionsBySymbol[sym]
			if !ok {
				continue
			}
			simBars := data.ToSimBars(bars)

			for _, lag := range cfg.Lags {
				for _, windowD := range cfg.WindowDays {
					windowH := windowD * 24
					bSlice := simBars
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

					jobs = append(jobs, SimJob{
						ID:      jobID,
						Bars:    bSlice,
						Actions: aSlice,
						Config:  simCfg,
					})
					metas = append(metas, jobMeta{Symbol: sym, Lag: lag, Window: windowD})
					jobID++
				}
			}
		}

		if len(jobs) == 0 {
			model.Close()
			continue
		}

		results := pool.Run(jobs)

		// Aggregate by (lag, window)
		type aggKey struct {
			Lag    int
			Window int
		}
		agg := make(map[aggKey]map[string]sim.SimResult)
		for i, r := range results {
			m := metas[i]
			key := aggKey{m.Lag, m.Window}
			if agg[key] == nil {
				agg[key] = make(map[string]sim.SimResult)
			}
			agg[key][m.Symbol] = r.Result
		}

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
			medSort := median(sorts)

			allResults = append(allResults, CheckpointResult{
				Name:        ckptName,
				Epoch:       ckptName,
				Lag:         key.Lag,
				WindowDays:  key.Window,
				MeanReturn:  meanRet * 100,
				MeanSortino: meanSort,
				MedianSort:  medSort,
				PosSym:      posSym,
				TotalSym:    len(symResults),
				TotalTrades: totalTrades,
				WorstDD:     worstDD * 100,
				PerSymbol:   perSym,
			})
		}

		model.Close()
	}

	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].MeanSortino > allResults[j].MeanSortino
	})

	return allResults, nil
}

func PrintLeaderboard(results []CheckpointResult) {
	fmt.Printf("\n%-30s %4s %3s %5s %8s %8s %5s %6s %7s\n",
		"Checkpoint", "Lag", "Win", "Pos", "MeanRet", "MeanSort", "MedS", "Trades", "WorstDD")
	fmt.Println(strings.Repeat("-", 90))
	for _, r := range results {
		fmt.Printf("%-30s %4d %3dd %d/%d  %+7.2f%% %8.2f %5.2f %6d %+6.1f%%\n",
			r.Name, r.Lag, r.WindowDays, r.PosSym, r.TotalSym,
			r.MeanReturn, r.MeanSortino, r.MedianSort, r.TotalTrades, r.WorstDD)
	}
}

func SaveCheckpointResults(results []CheckpointResult, path string) error {
	d, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, d, 0644)
}

func buildFeatureMatrix(bars []data.TimestampedBar, norm *data.FeatureNormalizer, seqLen int) [][]float32 {
	if len(bars) < seqLen {
		return nil
	}

	feats := data.ComputeFeatures(bars)
	featureNames := []string{"return_1h", "volatility_24h", "volume_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "high_low_range"}

	n := len(bars)
	result := make([][]float32, n)
	for i := 0; i < n; i++ {
		row := make([]float32, len(featureNames))
		for j, name := range featureNames {
			val := feats[name][i]
			if norm != nil {
				val = norm.Normalize(name, val)
			}
			row[j] = float32(val)
		}
		result[i] = row
	}
	return result
}

func median(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)
	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
}
