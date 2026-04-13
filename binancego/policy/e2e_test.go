package policy

import (
	"fmt"
	"os"
	"testing"

	"binancego/data"
	"binancego/sim"
)

func TestEndToEndONNXSim(t *testing.T) {
	modelPath := "/tmp/btc_epoch_001.onnx"
	csvPath := "/home/lee/code/stock/trainingdatahourly/crypto/BTCUSD.csv"
	ortLib := "/tmp/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so"

	for _, f := range []string{modelPath, csvPath, ortLib} {
		if _, err := os.Stat(f); err != nil {
			t.Skipf("required file not found: %s", f)
		}
	}

	if err := InitONNXRuntime(ortLib); err != nil {
		t.Fatalf("InitONNXRuntime: %v", err)
	}
	defer DestroyONNXRuntime()

	model, err := LoadONNX(modelPath)
	if err != nil {
		t.Fatalf("LoadONNX: %v", err)
	}
	defer model.Close()

	bars, err := data.LoadCSV(csvPath)
	if err != nil {
		t.Fatalf("LoadCSV: %v", err)
	}
	t.Logf("loaded %d bars", len(bars))

	// Use last 720 bars (30 days)
	seqLen := 48
	windowBars := 720
	if len(bars) > windowBars {
		bars = bars[len(bars)-windowBars:]
	}

	feats := data.ComputeFeatures(bars)
	featureNames := []string{"return_1h", "volatility_24h", "volume_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "high_low_range"}

	// Build feature matrix (need at least input_dim=16 for this model)
	// Pad with zeros if we have fewer than 16 features
	inputDim := 16
	features := make([][]float32, len(bars))
	for i := range bars {
		row := make([]float32, inputDim)
		for j, name := range featureNames {
			if j < inputDim {
				row[j] = float32(feats[name][i])
			}
		}
		features[i] = row
	}

	// Run inference in chunks of seqLen
	actions := make([]sim.Action, len(bars))
	cfg := DefaultDecodeConfig()

	for start := 0; start+seqLen <= len(bars); start += seqLen {
		chunk := features[start : start+seqLen]
		logits, err := model.InferSequence(chunk)
		if err != nil {
			t.Fatalf("InferSequence at %d: %v", start, err)
		}

		for i, logitRow := range logits {
			idx := start + i
			ref := bars[idx].Close
			high, low := ref*1.01, ref*0.99
			decoded := DecodeActions(logitRow, ref, high, low, cfg)
			actions[idx] = sim.Action{
				BuyPrice:   decoded.BuyPrice,
				SellPrice:  decoded.SellPrice,
				BuyAmount:  decoded.BuyAmount,
				SellAmount: decoded.SellAmount,
			}
		}
	}

	simBars := data.ToSimBars(bars)

	for _, lag := range []int{0, 1, 2} {
		bSlice, aSlice := sim.ApplyDecisionLag(simBars, actions, lag)
		if bSlice == nil {
			continue
		}

		simCfg := sim.SimConfig{
			MaxLeverage:    2.0,
			MakerFee:       0.001,
			InitialCash:    10000,
			FillBufferPct:  0.0005,
			MaxHoldBars:    6,
			IntensityScale: 1.0,
		}

		result := sim.Simulate(bSlice, aSlice, simCfg)
		t.Logf("lag=%d: bars=%d trades=%d ret=%+.4f%% sortino=%.2f dd=%.4f%%",
			lag, len(bSlice), result.NumTrades, result.TotalReturn*100,
			result.Sortino, result.MaxDrawdown*100)

		fmt.Printf("  E2E lag=%d: ret=%+.4f%% sortino=%.2f trades=%d dd=%.4f%%\n",
			lag, result.TotalReturn*100, result.Sortino, result.NumTrades, result.MaxDrawdown*100)
	}
}
