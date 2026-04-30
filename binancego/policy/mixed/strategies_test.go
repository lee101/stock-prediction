package mixed

import (
	"math"
	"testing"
)

func TestStrategyCount(t *testing.T) {
	got := len(LiveStrategies())
	const want = 36
	if got != want {
		t.Fatalf("LiveStrategies() = %d strategies, want %d (matches bitbankgo trading_bot.go:1112)", got, want)
	}
}

func TestStrategyExposureClamp(t *testing.T) {
	if got := StrategyExposure(LiveStrategy{Exposure: 0}); got != 1 {
		t.Errorf("Exposure=0 default => 1, got %v", got)
	}
	if got := StrategyExposure(LiveStrategy{Exposure: 0.10}); got != 0.25 {
		t.Errorf("Exposure=0.10 => clamped 0.25, got %v", got)
	}
	if got := StrategyExposure(LiveStrategy{Exposure: 5.0}); got != 1.25 {
		t.Errorf("Exposure=5 => clamped 1.25, got %v", got)
	}
}

func TestRebalanceHoursDefault(t *testing.T) {
	if got := RebalanceHours(LiveStrategy{RebalanceHrs: 0}); got != 1 {
		t.Errorf("RebalanceHrs=0 default => 1, got %v", got)
	}
	if got := RebalanceHours(LiveStrategy{RebalanceHrs: 6}); got != 6 {
		t.Errorf("RebalanceHrs=6 => 6, got %v", got)
	}
}

func TestExpectedReturnRev2H(t *testing.T) {
	// rev2h: 1 term, lookback=2 weight=-0.20.
	// candles: pad zeros for indexing safety, then closes that yield a known up-move.
	candles := make([]Candle, 5)
	for i := range candles {
		candles[i] = Candle{Close: 100.0}
	}
	candles[3].Close = 102.0
	candles[4].Close = 104.0 // 2-bar return = (104-100)/100 = 0.04, /Lookback=2 => 0.02
	got := StrategyExpectedReturn(candles, 4, LiveStrategy{Terms: []ReturnTerm{{Lookback: 2, Weight: -0.20}}})
	want := -0.20 * 0.02
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("rev2h expected_return = %v, want %v", got, want)
	}
}

func TestExpectedReturnClamped(t *testing.T) {
	candles := []Candle{{Close: 1}, {Close: 1}, {Close: 100}, {Close: 200}}
	// Wild return: 1->200 over 2 bars = 199/2 = 99.5 then *(-0.2) huge negative.
	got := StrategyExpectedReturn(candles, 3, LiveStrategy{Terms: []ReturnTerm{{Lookback: 2, Weight: -0.20}}})
	if math.Abs(got-(-0.02)) > 1e-12 {
		t.Errorf("expected clamp at -0.02, got %v", got)
	}
}

func TestSortinoAllPositive(t *testing.T) {
	// no negatives, mean>0 => returns 20 sentinel
	if got := sortinoRatio([]float64{0.01, 0.02, 0.03}); got != 20 {
		t.Errorf("all-pos sortino want 20, got %v", got)
	}
	// no negatives, mean<=0 => 0
	if got := sortinoRatio([]float64{0, 0, 0}); got != 0 {
		t.Errorf("zero-mean sortino want 0, got %v", got)
	}
}

func TestStrategyScoreLessIsZeroTrades(t *testing.T) {
	// Trades=0 => -inf so it never wins selection.
	got := StrategyScore(Performance{Trades: 0, PnLPct: 100, Sortino: 5})
	if !math.IsInf(got, -1) {
		t.Errorf("Trades=0 score want -Inf, got %v", got)
	}
}

func TestBuyAndHoldSimpleUp(t *testing.T) {
	candles := make([]Candle, 30)
	price := 100.0
	for i := range candles {
		candles[i] = Candle{Open: price, High: price, Low: price, Close: price}
		price *= 1.01
	}
	perf, ok := BuyAndHoldPerformance(candles, 0, len(candles)-1, 0)
	if !ok {
		t.Fatal("BuyAndHoldPerformance unexpectedly returned false")
	}
	want := math.Pow(1.01, float64(len(candles)-1)) - 1 // ≈ 1.01^29 - 1
	got := perf.PnLPct / 100
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("buy-and-hold PnLPct = %v, want %v", got, want)
	}
}

func TestSimulateLimitBacktestEmptyCandles(t *testing.T) {
	res := SimulateLimitBacktest("X", nil, 0, 0, LiveStrategy{Name: "rev2h"}, 1, 0, 0, false)
	if res.Perf.Trades != 0 {
		t.Errorf("empty candles want 0 trades, got %v", res.Perf.Trades)
	}
}

func TestMixedCandidateScoreBaseline(t *testing.T) {
	// pnl=10, dd=2, sortino=1.5, trades=4 => 10*1.12 - 2*0.65 + 1.5*2.05 - 2*0.06
	got := MixedCandidateScore(10, 2, 1.5, 4)
	want := 10*1.12 - 2*0.65 + 1.5*2.05 - math.Sqrt(4)*0.06
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("MixedCandidateScore=%v, want %v", got, want)
	}
}

func TestXGBCandidateBonusOff(t *testing.T) {
	// trades<8 → 0
	if got := XGBCandidateBonus(PairSignal{XGBTrades30D: 5, XGBSortino30D: 2, XGBPnL30DPct: 10}); got != 0 {
		t.Errorf("low trades => 0, got %v", got)
	}
	// sortino<=0 → 0
	if got := XGBCandidateBonus(PairSignal{XGBTrades30D: 10, XGBSortino30D: 0, XGBPnL30DPct: 10}); got != 0 {
		t.Errorf("zero sortino => 0, got %v", got)
	}
	// pnl<=0 → 0
	if got := XGBCandidateBonus(PairSignal{XGBTrades30D: 10, XGBSortino30D: 1, XGBPnL30DPct: 0}); got != 0 {
		t.Errorf("zero pnl => 0, got %v", got)
	}
}

func TestMixSignalEmpty(t *testing.T) {
	if got := MixSignal(nil); got != nil {
		t.Errorf("empty signals want nil, got %+v", got)
	}
}

func TestMixSignalBuyMajority(t *testing.T) {
	// 4 buys, 1 sell, mid prices fine. Expect SignalType="buy".
	mk := func(pair, side string, pnl, sort float64) PairSignal {
		return PairSignal{
			Pair: pair, SignalType: side,
			BuyPrice: 99.91, SellPrice: 100.09,
			PnL7DPct: pnl, MaxDD7DPct: 2.0, Sortino7D: sort, Trades7D: 10,
			Confidence: 0.6, RebalanceHours: 1,
		}
	}
	signals := []PairSignal{
		mk("AAA", "buy", 5, 1.5),
		mk("BBB", "buy", 4, 1.4),
		mk("CCC", "buy", 6, 1.6),
		mk("DDD", "buy", 3, 1.2),
		mk("EEE", "sell", 2, 0.5),
	}
	dec := MixSignal(signals)
	if dec == nil {
		t.Fatal("decision unexpectedly nil")
	}
	if dec.SignalType != "buy" {
		t.Errorf("got SignalType=%v, want buy", dec.SignalType)
	}
	if len(dec.Constituents) < 3 {
		t.Errorf("expected at least 3 constituents, got %d", len(dec.Constituents))
	}
}

func TestSimulatePortfolioFlat(t *testing.T) {
	// Constant-price symbol over 200 bars. Strategy can't make money — expect ~0% return.
	bars := make([]Candle, 200)
	for i := range bars {
		bars[i] = Candle{Open: 100, High: 100.01, Low: 99.99, Close: 100, Timestamp: int64(i) * 3600}
	}
	syms := []SymbolBars{
		{Symbol: "AAA", Bars: bars},
		{Symbol: "BBB", Bars: bars},
		{Symbol: "CCC", Bars: bars},
	}
	cfg := DefaultPortfolioConfig()
	cfg.MaxLeverage = 1.0
	res := SimulatePortfolio(syms, 100, 199, cfg)
	if math.Abs(res.TotalReturn) > 0.01 {
		t.Errorf("flat market portfolio return = %v, want ~0", res.TotalReturn)
	}
}
