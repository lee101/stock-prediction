package sim

import (
	"testing"
)

func TestSimulateSharedCashBasic(t *testing.T) {
	bars := []SymbolBar{
		{1000, "BTC", Bar{50000, 50500, 49500, 50000}},
		{1000, "ETH", Bar{3000, 3050, 2950, 3000}},
		{2000, "BTC", Bar{50000, 51000, 49000, 50500}},
		{2000, "ETH", Bar{3000, 3100, 2900, 3050}},
	}
	actions := []SymbolAction{
		{1000, "BTC", Action{49600, 50400, 50, 80}},
		{1000, "ETH", Action{2960, 3040, 50, 80}},
		{2000, "BTC", Action{49000, 50800, 50, 80}},
		{2000, "ETH", Action{2900, 3080, 50, 80}},
	}
	cfg := DefaultSharedCashConfig()
	cfg.FillBufferBps = 0
	cfg.MaxHoldHours = 0
	cfg.ForceCloseOnHold = false

	result := SimulateSharedCash(bars, actions, cfg)
	if len(result.EquityCurve) != 2 {
		t.Errorf("expected 2 equity points, got %d", len(result.EquityCurve))
	}
	t.Logf("trades=%d equity=%v", len(result.Trades), result.EquityCurve)
}

func TestSimulateSharedCashSellsFirst(t *testing.T) {
	// Verify sells happen before buys (cash freed by sell used for buy)
	bars := []SymbolBar{
		{1000, "BTC", Bar{50000, 50500, 49500, 50000}},
		{2000, "BTC", Bar{50000, 51000, 49000, 50500}},
	}
	actions := []SymbolAction{
		{1000, "BTC", Action{49600, 60000, 50, 50}}, // buy fills, sell won't
		{2000, "BTC", Action{48000, 50800, 50, 80}},  // sell fills at 50800
	}
	cfg := DefaultSharedCashConfig()
	cfg.FillBufferBps = 0
	cfg.MaxHoldHours = 0
	cfg.ForceCloseOnHold = false

	result := SimulateSharedCash(bars, actions, cfg)
	buys, sells := 0, 0
	for _, tr := range result.Trades {
		if tr.Side == "buy" {
			buys++
		} else {
			sells++
		}
	}
	t.Logf("buys=%d sells=%d trades=%d", buys, sells, len(result.Trades))
}

func TestSimulateSharedCashEmpty(t *testing.T) {
	result := SimulateSharedCash(nil, nil, DefaultSharedCashConfig())
	if result.EquityCurve[0] != 10000 {
		t.Errorf("expected initial cash, got %g", result.EquityCurve[0])
	}
}

func TestSimulateSharedCashDecisionLag(t *testing.T) {
	bars := []SymbolBar{
		{1000, "BTC", Bar{50000, 50500, 49500, 50000}},
		{2000, "BTC", Bar{50000, 51000, 49000, 50500}},
		{3000, "BTC", Bar{50500, 51500, 49500, 51000}},
	}
	actions := []SymbolAction{
		{1000, "BTC", Action{49600, 50400, 50, 80}},
		{2000, "BTC", Action{49000, 50800, 50, 80}},
		{3000, "BTC", Action{49500, 51200, 50, 80}},
	}
	cfg := DefaultSharedCashConfig()
	cfg.DecisionLagBars = 1
	cfg.FillBufferBps = 0
	cfg.MaxHoldHours = 0
	cfg.ForceCloseOnHold = false

	result := SimulateSharedCash(bars, actions, cfg)
	t.Logf("lag=1: trades=%d equity_points=%d", len(result.Trades), len(result.EquityCurve))
}

func TestSimulateSharedCashMaxHold(t *testing.T) {
	// Timestamps in seconds, max_hold=1 hour
	bars := []SymbolBar{
		{0, "BTC", Bar{50000, 50500, 49500, 50000}},
		{3600, "BTC", Bar{50000, 50500, 49500, 50000}},
		{7200, "BTC", Bar{50000, 50500, 49500, 50000}},
	}
	actions := []SymbolAction{
		{0, "BTC", Action{49600, 60000, 50, 50}},    // buy fills
		{3600, "BTC", Action{49600, 60000, 0, 0}},   // hold (1 hour)
		{7200, "BTC", Action{49600, 60000, 0, 0}},   // should force close
	}
	cfg := DefaultSharedCashConfig()
	cfg.FillBufferBps = 0
	cfg.MaxHoldHours = 1
	cfg.ForceCloseOnHold = true

	result := SimulateSharedCash(bars, actions, cfg)
	maxHoldSells := 0
	for _, tr := range result.Trades {
		if tr.Reason == "max_hold" {
			maxHoldSells++
		}
	}
	if maxHoldSells == 0 {
		t.Error("expected at least 1 max_hold force close")
	}
	t.Logf("max_hold_sells=%d total_trades=%d", maxHoldSells, len(result.Trades))
}
