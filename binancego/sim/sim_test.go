package sim

import (
	"math"
	"testing"
)

func TestSimulateNoTrades(t *testing.T) {
	bars := []Bar{
		{100, 105, 95, 102},
		{102, 108, 98, 104},
	}
	// actions with 0 amounts => no trades
	actions := []Action{
		{98, 106, 0, 0},
		{99, 107, 0, 0},
	}
	cfg := DefaultSimConfig()
	r := Simulate(bars, actions, cfg)
	if r.NumTrades != 0 {
		t.Errorf("expected 0 trades, got %d", r.NumTrades)
	}
	if r.FinalEquity != cfg.InitialCash {
		t.Errorf("equity should be unchanged: %g != %g", r.FinalEquity, cfg.InitialCash)
	}
}

func TestSimulateBuyAndSell(t *testing.T) {
	// Bar 0: low=95 hits buy at 96 (with 0 buffer), buys
	// Bar 1: high=110 hits sell at 108 (with 0 buffer), sells
	bars := []Bar{
		{100, 105, 95, 100},
		{101, 110, 99, 105},
	}
	actions := []Action{
		{96, 200, 100, 100},  // buy at 96, sell way above (won't fill bar 0)
		{80, 108, 100, 100},  // buy way below (won't fill bar 1), sell at 108
	}
	cfg := DefaultSimConfig()
	cfg.FillBufferPct = 0
	cfg.MaxLeverage = 10
	r := Simulate(bars, actions, cfg)
	if r.NumTrades != 2 {
		t.Fatalf("expected 2 trades, got %d", r.NumTrades)
	}
	if r.TotalReturn <= 0 {
		t.Errorf("expected positive return, got %g", r.TotalReturn)
	}
}

func TestSimulateForceClose(t *testing.T) {
	bars := []Bar{
		{100, 105, 90, 100},
		{100, 105, 95, 100},
		{100, 105, 95, 100},
	}
	actions := []Action{
		{95, 200, 100, 100},   // buy
		{80, 200, 0, 0},       // hold
		{80, 200, 0, 0},       // force close
	}
	cfg := DefaultSimConfig()
	cfg.FillBufferPct = 0
	cfg.MaxLeverage = 10
	cfg.MaxHoldBars = 2
	r := Simulate(bars, actions, cfg)
	// Should have 2 trades: 1 buy + 1 force close
	if r.NumTrades != 2 {
		t.Errorf("expected 2 trades (buy + force close), got %d", r.NumTrades)
	}
}

func TestSimulateMarginInterest(t *testing.T) {
	// Buy more than cash allows with leverage -> negative cash -> margin interest
	bars := []Bar{
		{100, 105, 90, 100},
		{100, 105, 95, 100},
	}
	actions := []Action{
		{95, 200, 100, 100},
		{80, 200, 0, 100}, // sell
	}
	cfg := DefaultSimConfig()
	cfg.FillBufferPct = 0
	cfg.MaxLeverage = 3
	cfg.MarginHourlyRate = 0.001
	r := Simulate(bars, actions, cfg)
	if r.MarginCostTotal <= 0 {
		t.Errorf("expected margin cost > 0, got %g", r.MarginCostTotal)
	}
}

func TestSimulateNoSameBarRoundtrip(t *testing.T) {
	// If sell fills, buy should NOT fill on same bar
	bars := []Bar{
		{100, 120, 80, 100}, // wide range, both could fill
	}
	actions := []Action{
		{85, 115, 100, 100}, // both buy and sell could fill
	}
	cfg := DefaultSimConfig()
	cfg.FillBufferPct = 0
	cfg.MaxLeverage = 10
	// No inventory to sell, so buy should fill
	r := Simulate(bars, actions, cfg)
	if r.NumTrades != 1 {
		t.Errorf("expected 1 trade (buy only, no roundtrip), got %d", r.NumTrades)
	}
}

func TestSimulateMinEdge(t *testing.T) {
	bars := []Bar{
		{100, 105, 95, 100},
	}
	// edge = (101 - 99) / 99 = ~0.02
	actions := []Action{
		{99, 101, 100, 100},
	}
	cfg := DefaultSimConfig()
	cfg.FillBufferPct = 0
	cfg.MinEdge = 0.05 // require 5% edge, only ~2% available
	r := Simulate(bars, actions, cfg)
	if r.NumTrades != 0 {
		t.Errorf("expected 0 trades (min edge filter), got %d", r.NumTrades)
	}
}

func TestApplyDecisionLag(t *testing.T) {
	bars := make([]Bar, 5)
	actions := make([]Action, 5)
	for i := range bars {
		bars[i] = Bar{float64(i), float64(i), float64(i), float64(i)}
		actions[i] = Action{float64(i * 10), float64(i * 10), 1, 1}
	}

	bLag, aLag := ApplyDecisionLag(bars, actions, 2)
	if len(bLag) != 3 || len(aLag) != 3 {
		t.Fatalf("expected 3 bars/actions, got %d/%d", len(bLag), len(aLag))
	}
	// bars should start at index 2
	if bLag[0].Close != 2 {
		t.Errorf("expected bars[0].Close=2, got %g", bLag[0].Close)
	}
	// actions should be from original index 0
	if aLag[0].BuyPrice != 0 {
		t.Errorf("expected actions[0].BuyPrice=0, got %g", aLag[0].BuyPrice)
	}
}

func TestApplyDecisionLagZero(t *testing.T) {
	bars := []Bar{{1, 2, 0, 1}}
	actions := []Action{{1, 2, 1, 1}}
	b, a := ApplyDecisionLag(bars, actions, 0)
	if len(b) != 1 || len(a) != 1 {
		t.Errorf("lag=0 should return same slices")
	}
}

func TestSimulateBatch(t *testing.T) {
	bars := []Bar{
		{100, 105, 95, 102},
		{102, 108, 98, 104},
	}
	actions := []Action{
		{96, 106, 100, 100},
		{97, 107, 100, 100},
	}
	configs := []SimConfig{
		DefaultSimConfig(),
		DefaultSimConfig(),
	}
	configs[1].MakerFee = 0.01
	results := SimulateBatch(bars, actions, configs)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
}

func TestSimulateEmpty(t *testing.T) {
	r := Simulate(nil, nil, DefaultSimConfig())
	if r.FinalEquity != 10000 {
		t.Errorf("empty sim should return initial cash")
	}
}

func TestSortinoMatchesCSimOnSameData(t *testing.T) {
	// Verify Go Sortino on an equity curve.
	// Using the CANONICAL formula: sqrt(sum_neg_sq / total_n).
	eq := []float64{10000, 10100, 9950, 10200, 10050, 10300, 10150, 10400}
	returns := StepReturns(eq)

	var sumRet, sumNegSq float64
	for _, r := range returns {
		sumRet += r
		if r < 0 {
			sumNegSq += r * r
		}
	}
	nRet := float64(len(returns))
	meanRet := sumRet / nRet
	downsideDev := math.Sqrt(sumNegSq / nRet)
	expected := (meanRet / downsideDev) * math.Sqrt(8760)

	got := Sortino(eq, 8760)
	if !almostEqual(got, expected, 1e-10) {
		t.Errorf("Sortino = %.10f, expected %.10f", got, expected)
	}
}
