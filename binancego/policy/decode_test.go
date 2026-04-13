package policy

import (
	"math"
	"testing"
)

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

func TestSigmoid(t *testing.T) {
	if !almostEqual(sigmoid(0), 0.5, 1e-10) {
		t.Errorf("sigmoid(0) = %g, want 0.5", sigmoid(0))
	}
	if sigmoid(100) < 0.999 {
		t.Errorf("sigmoid(100) should be ~1, got %g", sigmoid(100))
	}
	if sigmoid(-100) > 0.001 {
		t.Errorf("sigmoid(-100) should be ~0, got %g", sigmoid(-100))
	}
}

func TestDecodeActionsMidpoint(t *testing.T) {
	cfg := DefaultDecodeConfig()
	cfg.UseMidpointOffsets = true

	// logits = [0, 0, 0, 0] -> sigmoid = [0.5, 0.5, 0.5, 0.5]
	logits := []float64{0, 0, 0, 0}
	refClose := 50000.0
	chronosHigh := 51000.0
	chronosLow := 49000.0

	a := DecodeActions(logits, refClose, chronosHigh, chronosLow, cfg)

	// buyUnit=0.5: buy = 49000 + 0.5*(50000-49000) = 49500
	if !almostEqual(a.BuyPrice, 49500, 1) {
		t.Errorf("BuyPrice = %g, want ~49500", a.BuyPrice)
	}
	// sellUnit=0.5: sell = 50000 + 0.5*(51000-50000) = 50500
	if !almostEqual(a.SellPrice, 50500, 1) {
		t.Errorf("SellPrice = %g, want ~50500", a.SellPrice)
	}
	// amounts: sigmoid(0) * 100 = 50
	if !almostEqual(a.BuyAmount, 50, 0.1) {
		t.Errorf("BuyAmount = %g, want ~50", a.BuyAmount)
	}
}

func TestDecodeActionsGapEnforcement(t *testing.T) {
	cfg := DefaultDecodeConfig()
	cfg.UseMidpointOffsets = true
	cfg.MinGapPct = 0.01

	// Very negative logits -> buy and sell both near ref
	logits := []float64{-10, -10, 0, 0}
	refClose := 50000.0
	a := DecodeActions(logits, refClose, 50000, 50000, cfg)

	gap := refClose * cfg.MinGapPct
	if a.SellPrice < a.BuyPrice+gap-0.01 {
		t.Errorf("gap not enforced: sell=%g buy=%g gap=%g", a.SellPrice, a.BuyPrice, gap)
	}
}

func TestDecodeActionsOffset(t *testing.T) {
	cfg := DefaultDecodeConfig()
	cfg.UseMidpointOffsets = false
	cfg.PriceOffsetPct = 0.02

	logits := []float64{0, 0, 0, 0} // sigmoid=0.5 for all
	refClose := 50000.0

	a := DecodeActions(logits, refClose, 51000, 49000, cfg)

	// buy = 50000 * (1 - 0.02*0.5) = 50000 * 0.99 = 49500
	if !almostEqual(a.BuyPrice, 49500, 1) {
		t.Errorf("BuyPrice = %g, want ~49500", a.BuyPrice)
	}
	// sell = 50000 * (1 + 0.02*0.5) = 50000 * 1.01 = 50500
	if !almostEqual(a.SellPrice, 50500, 1) {
		t.Errorf("SellPrice = %g, want ~50500", a.SellPrice)
	}
}

func TestDecodeActionsBatch(t *testing.T) {
	cfg := DefaultDecodeConfig()
	logits := [][]float64{
		{0, 0, 0, 0},
		{1, 1, 1, 1},
		{-1, -1, -1, -1},
	}
	refs := []float64{50000, 50000, 50000}
	highs := []float64{51000, 51000, 51000}
	lows := []float64{49000, 49000, 49000}

	actions := DecodeActionsBatch(logits, refs, highs, lows, cfg)
	if len(actions) != 3 {
		t.Fatalf("expected 3 actions, got %d", len(actions))
	}
	// Higher logits -> higher sigmoid -> more aggressive amounts
	if actions[1].BuyAmount <= actions[0].BuyAmount {
		t.Errorf("logits[1] should produce larger amounts than logits[0]")
	}
	if actions[2].BuyAmount >= actions[0].BuyAmount {
		t.Errorf("logits[-1] should produce smaller amounts than logits[0]")
	}
}

func TestDecodeActionsWithHoldHours(t *testing.T) {
	cfg := DefaultDecodeConfig()
	logits := []float64{0, 0, 0, 0, 0} // 5th element = hold hours
	a := DecodeActions(logits, 50000, 51000, 49000, cfg)
	// sigmoid(0) * 24 = 12
	if !almostEqual(a.HoldHours, 12, 0.1) {
		t.Errorf("HoldHours = %g, want ~12", a.HoldHours)
	}
}
