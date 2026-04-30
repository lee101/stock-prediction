package loader

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadStockCandlesAAPL(t *testing.T) {
	path := "/nvme0n1-disk/code/stock-prediction/trainingdatahourly/stocks/AAPL.csv"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("AAPL.csv not present at %s: %v", path, err)
	}
	candles, err := LoadStockCandles(path)
	if err != nil {
		t.Fatalf("LoadStockCandles: %v", err)
	}
	if len(candles) < 100 {
		t.Fatalf("expected hundreds of bars, got %d", len(candles))
	}
	for i, c := range candles {
		if c.Close <= 0 || c.High < c.Low {
			t.Fatalf("bar %d invalid: %+v", i, c)
		}
	}
	// Sorted ascending in time.
	for i := 1; i < len(candles); i++ {
		if candles[i].Timestamp < candles[i-1].Timestamp {
			t.Fatalf("non-monotonic at %d", i)
		}
	}
}

func TestLoadAlignedStockCandles(t *testing.T) {
	root := "/nvme0n1-disk/code/stock-prediction/trainingdatahourly/stocks"
	if _, err := os.Stat(filepath.Join(root, "AAPL.csv")); err != nil {
		t.Skipf("AAPL.csv missing: %v", err)
	}
	syms, err := LoadAlignedStockCandles(root, []string{"AAPL", "MSFT", "NVDA"})
	if err != nil {
		t.Fatalf("LoadAlignedStockCandles: %v", err)
	}
	if len(syms) != 3 {
		t.Fatalf("want 3 symbols, got %d", len(syms))
	}
	n := len(syms[0].Bars)
	for _, s := range syms {
		if len(s.Bars) != n {
			t.Fatalf("aligned lengths differ: %s=%d vs %d", s.Symbol, len(s.Bars), n)
		}
	}
	// Spot-check timestamp alignment across symbols.
	for i := 0; i < n; i++ {
		ts := syms[0].Bars[i].Timestamp
		for _, s := range syms[1:] {
			if s.Bars[i].Timestamp != ts {
				t.Fatalf("ts mismatch at idx %d: %s=%d vs %d", i, s.Symbol, s.Bars[i].Timestamp, ts)
			}
		}
	}
}
