package sim

import (
	"math"
	"testing"
)

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

func TestStepReturns(t *testing.T) {
	eq := []float64{100, 110, 105, 115}
	ret := StepReturns(eq)
	expected := []float64{0.1, -0.04545454545454545, 0.09523809523809523}
	if len(ret) != len(expected) {
		t.Fatalf("len %d != %d", len(ret), len(expected))
	}
	for i := range ret {
		if !almostEqual(ret[i], expected[i], 1e-10) {
			t.Errorf("ret[%d] = %g, want %g", i, ret[i], expected[i])
		}
	}
}

func TestStepReturnsEmpty(t *testing.T) {
	if r := StepReturns(nil); r != nil {
		t.Errorf("expected nil, got %v", r)
	}
	if r := StepReturns([]float64{100}); r != nil {
		t.Errorf("expected nil, got %v", r)
	}
}

func TestMaxDrawdown(t *testing.T) {
	eq := []float64{100, 110, 90, 95, 105, 80}
	dd := MaxDrawdown(eq)
	// peak=110, trough=80 -> dd = (80-110)/110 = -0.2727...
	expected := (80.0 - 110.0) / 110.0
	if !almostEqual(dd, expected, 1e-6) {
		t.Errorf("MaxDrawdown = %g, want %g", dd, expected)
	}
}

func TestMaxDrawdownNoDD(t *testing.T) {
	eq := []float64{100, 110, 120, 130}
	dd := MaxDrawdown(eq)
	if dd != 0 {
		t.Errorf("expected 0, got %g", dd)
	}
}

func TestSortinoCanonical(t *testing.T) {
	// Hand-computed example matching metrics_utils.py formula.
	// equity: [100, 102, 99, 103, 101, 104]
	// returns: [0.02, -0.0294117647, 0.0404040404, -0.0194174757, 0.0297029703]
	eq := []float64{100, 102, 99, 103, 101, 104}
	returns := StepReturns(eq)

	var sumRet, sumNegSq float64
	for _, r := range returns {
		sumRet += r
		if r < 0 {
			sumNegSq += r * r
		}
	}
	meanRet := sumRet / float64(len(returns))
	downsideDev := math.Sqrt(sumNegSq / float64(len(returns)))
	expectedSortino := (meanRet / downsideDev) * math.Sqrt(8760)

	gotSortino := Sortino(eq, 8760)
	if !almostEqual(gotSortino, expectedSortino, 1e-8) {
		t.Errorf("Sortino = %g, want %g", gotSortino, expectedSortino)
	}
}

func TestSortinoAllPositive(t *testing.T) {
	eq := []float64{100, 101, 102, 103}
	s := Sortino(eq, 8760)
	// Should fall back to Sharpe (no negative returns)
	if s <= 0 {
		t.Errorf("expected positive Sortino for all-positive returns, got %g", s)
	}
}

func TestSortinoTooShort(t *testing.T) {
	if s := Sortino([]float64{100, 101}, 8760); s != 0 {
		t.Errorf("expected 0 for 2-point curve, got %g", s)
	}
}
