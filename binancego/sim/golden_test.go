package sim

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type goldenData struct {
	Bars    []struct{ Open, High, Low, Close float64 } `json:"bars"`
	Actions []struct {
		BuyPrice  float64 `json:"buy_price"`
		SellPrice float64 `json:"sell_price"`
		BuyAmount float64 `json:"buy_amount"`
		SellAmount float64 `json:"sell_amount"`
	} `json:"actions"`
	Config struct {
		MaxLeverage      float64 `json:"max_leverage"`
		CanShort         bool    `json:"can_short"`
		MakerFee         float64 `json:"maker_fee"`
		MarginHourlyRate float64 `json:"margin_hourly_rate"`
		InitialCash      float64 `json:"initial_cash"`
		FillBufferPct    float64 `json:"fill_buffer_pct"`
		MinEdge          float64 `json:"min_edge"`
		MaxHoldBars      int     `json:"max_hold_bars"`
		IntensityScale   float64 `json:"intensity_scale"`
	} `json:"config"`
	Expected struct {
		TotalReturn     float64   `json:"total_return"`
		Sortino         float64   `json:"sortino"`
		MaxDrawdown     float64   `json:"max_drawdown"`
		FinalEquity     float64   `json:"final_equity"`
		NumTrades       int       `json:"num_trades"`
		MarginCostTotal float64   `json:"margin_cost_total"`
		EquityCurve     []float64 `json:"equity_curve"`
	} `json:"expected"`
}

func TestGoldenSimulation(t *testing.T) {
	data, err := os.ReadFile("../internal/testdata/golden_sim.json")
	if err != nil {
		t.Skipf("golden test data not found: %v", err)
	}

	var g goldenData
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatalf("parse golden data: %v", err)
	}

	bars := make([]Bar, len(g.Bars))
	for i, b := range g.Bars {
		bars[i] = Bar{b.Open, b.High, b.Low, b.Close}
	}
	actions := make([]Action, len(g.Actions))
	for i, a := range g.Actions {
		actions[i] = Action{a.BuyPrice, a.SellPrice, a.BuyAmount, a.SellAmount}
	}

	cfg := SimConfig{
		MaxLeverage:      g.Config.MaxLeverage,
		CanShort:         g.Config.CanShort,
		MakerFee:         g.Config.MakerFee,
		MarginHourlyRate: g.Config.MarginHourlyRate,
		InitialCash:      g.Config.InitialCash,
		FillBufferPct:    g.Config.FillBufferPct,
		MinEdge:          g.Config.MinEdge,
		MaxHoldBars:      g.Config.MaxHoldBars,
		IntensityScale:   g.Config.IntensityScale,
	}

	result := Simulate(bars, actions, cfg)

	tol := 1e-6
	if !almostEqual(result.TotalReturn, g.Expected.TotalReturn, tol) {
		t.Errorf("TotalReturn: got %g, want %g", result.TotalReturn, g.Expected.TotalReturn)
	}
	if result.NumTrades != g.Expected.NumTrades {
		t.Errorf("NumTrades: got %d, want %d", result.NumTrades, g.Expected.NumTrades)
	}
	if !almostEqual(result.FinalEquity, g.Expected.FinalEquity, 0.01) {
		t.Errorf("FinalEquity: got %g, want %g", result.FinalEquity, g.Expected.FinalEquity)
	}
	if !almostEqual(result.MaxDrawdown, g.Expected.MaxDrawdown, tol) {
		t.Errorf("MaxDrawdown: got %g, want %g", result.MaxDrawdown, g.Expected.MaxDrawdown)
	}
	if !almostEqual(result.MarginCostTotal, g.Expected.MarginCostTotal, tol) {
		t.Errorf("MarginCostTotal: got %g, want %g", result.MarginCostTotal, g.Expected.MarginCostTotal)
	}

	// Validate equity curve point-by-point
	if len(result.EquityCurve) != len(g.Expected.EquityCurve) {
		t.Fatalf("EquityCurve length: got %d, want %d", len(result.EquityCurve), len(g.Expected.EquityCurve))
	}
	maxEqDiff := 0.0
	for i := range result.EquityCurve {
		diff := math.Abs(result.EquityCurve[i] - g.Expected.EquityCurve[i])
		if diff > maxEqDiff {
			maxEqDiff = diff
		}
	}
	if maxEqDiff > 0.01 {
		t.Errorf("max equity curve diff: %g (want < 0.01)", maxEqDiff)
	}

	// Sortino: compare with separate tolerance since formula variants exist
	if !almostEqual(result.Sortino, g.Expected.Sortino, 0.1) {
		t.Errorf("Sortino: got %g, want %g (tolerance 0.1)", result.Sortino, g.Expected.Sortino)
	}
	t.Logf("Golden results: ret=%.4f%% sort=%.2f dd=%.4f%% trades=%d eq_max_diff=%.6f",
		result.TotalReturn*100, result.Sortino, result.MaxDrawdown*100, result.NumTrades, maxEqDiff)
}
