package sim

import (
	"math"
	"sort"
)

// SymbolBar is a bar with symbol and timestamp info.
type SymbolBar struct {
	Timestamp int64
	Symbol    string
	Bar
}

// SymbolAction is an action with symbol and timestamp info.
type SymbolAction struct {
	Timestamp int64
	Symbol    string
	Action
}

// SimulateSharedCash runs a multi-symbol simulation with shared cash,
// matching marketsimulator.py:run_shared_cash_simulation().
func SimulateSharedCash(bars []SymbolBar, actions []SymbolAction, cfg SharedCashConfig) SharedCashResult {
	if len(bars) == 0 || len(actions) == 0 {
		return SharedCashResult{
			EquityCurve: []float64{cfg.InitialCash},
			Metrics:     map[string]float64{"total_return": 0, "sortino": 0},
		}
	}

	// Apply decision lag
	if cfg.DecisionLagBars > 0 {
		actions = applySharedLag(bars, actions, cfg.DecisionLagBars)
	}

	// Merge bars and actions by (timestamp, symbol)
	type mergedRow struct {
		Timestamp int64
		Symbol    string
		Bar
		Action
		HasAction bool
	}

	// Index actions by (timestamp, symbol)
	actionIdx := make(map[[2]interface{}]Action)
	for _, a := range actions {
		key := [2]interface{}{a.Timestamp, a.Symbol}
		actionIdx[key] = a.Action
	}

	var merged []mergedRow
	for _, b := range bars {
		key := [2]interface{}{b.Timestamp, b.Symbol}
		if a, ok := actionIdx[key]; ok {
			merged = append(merged, mergedRow{
				Timestamp: b.Timestamp, Symbol: b.Symbol,
				Bar: b.Bar, Action: a, HasAction: true,
			})
		}
	}

	if len(merged) == 0 {
		return SharedCashResult{
			EquityCurve: []float64{cfg.InitialCash},
			Metrics:     map[string]float64{"total_return": 0, "sortino": 0},
		}
	}

	// Sort by (timestamp, symbol)
	sort.Slice(merged, func(i, j int) bool {
		if merged[i].Timestamp != merged[j].Timestamp {
			return merged[i].Timestamp < merged[j].Timestamp
		}
		return merged[i].Symbol < merged[j].Symbol
	})

	// Group by timestamp
	type tsGroup struct {
		Timestamp int64
		Rows      []mergedRow
	}
	var groups []tsGroup
	for i := 0; i < len(merged); {
		ts := merged[i].Timestamp
		j := i
		for j < len(merged) && merged[j].Timestamp == ts {
			j++
		}
		groups = append(groups, tsGroup{Timestamp: ts, Rows: merged[i:j]})
		i = j
	}

	cash := cfg.InitialCash
	inventory := make(map[string]float64)
	costBasis := make(map[string]float64)
	openTime := make(map[string]int64) // timestamp when position opened
	var equityCurve []float64
	var trades []TradeRecord
	buffer := cfg.FillBufferBps / 10000.0

	for _, grp := range groups {
		ts := grp.Timestamp

		// Phase 1: sells first (release cash)
		for _, row := range grp.Rows {
			sym := row.Symbol
			inv := inventory[sym]
			if inv <= 0 {
				continue
			}
			sp := quantizeUp(row.SellPrice, cfg.TickSize)
			sa := row.SellAmount
			if sa <= 0 {
				continue
			}
			sellIntensity := clamp(sa/resolveScale(sa), 0, 1)
			sellFill := sp > 0 && row.High >= sp*(1+buffer) && sellIntensity > 0
			if !sellFill {
				continue
			}
			execSell := sellIntensity * inv
			execSell = quantizeDown(execSell, cfg.StepSize)
			if cfg.MinQty > 0 && execSell < cfg.MinQty {
				continue
			}
			if cfg.MinNotional > 0 && execSell*sp < cfg.MinNotional {
				continue
			}
			if execSell <= 0 {
				continue
			}
			proceeds := execSell * sp * (1 - cfg.MakerFee)
			cash += proceeds
			realized := (sp - costBasis[sym]) * execSell
			inv -= execSell
			if inv <= 0 {
				inv = 0
				costBasis[sym] = 0
				delete(openTime, sym)
			}
			inventory[sym] = inv
			trades = append(trades, TradeRecord{
				Timestamp: ts, Symbol: sym, Side: "sell",
				Price: sp, Quantity: execSell, Notional: execSell * sp,
				Fee: execSell * sp * cfg.MakerFee,
				CashAfter: cash, InventoryAfter: inv, RealizedPnl: realized,
				Reason: "signal",
			})
		}

		// Phase 2: buys (consume remaining cash)
		for _, row := range grp.Rows {
			sym := row.Symbol
			bp := quantizeDown(row.BuyPrice, cfg.TickSize)
			ba := row.BuyAmount
			if ba <= 0 {
				continue
			}
			buyIntensity := clamp(ba/resolveScale(ba), 0, 1)
			buyFill := bp > 0 && row.Low <= bp*(1-buffer) && buyIntensity > 0
			if !buyFill || cash <= 0 {
				continue
			}
			maxBuy := cash / (bp * (1 + cfg.MakerFee))
			execBuy := buyIntensity * maxBuy
			execBuy = quantizeDown(execBuy, cfg.StepSize)
			if cfg.MinQty > 0 && execBuy < cfg.MinQty {
				continue
			}
			if cfg.MinNotional > 0 && execBuy*bp < cfg.MinNotional {
				continue
			}
			if execBuy <= 0 {
				continue
			}
			cost := execBuy * bp * (1 + cfg.MakerFee)
			cash -= cost
			inv := inventory[sym]
			if inv <= 0 {
				costBasis[sym] = bp
				openTime[sym] = ts
			} else {
				costBasis[sym] = (costBasis[sym]*inv + bp*execBuy) / (inv + execBuy)
			}
			inventory[sym] = inv + execBuy
			trades = append(trades, TradeRecord{
				Timestamp: ts, Symbol: sym, Side: "buy",
				Price: bp, Quantity: execBuy, Notional: execBuy * bp,
				Fee: execBuy * bp * cfg.MakerFee,
				CashAfter: cash, InventoryAfter: inventory[sym], RealizedPnl: 0,
				Reason: "signal",
			})
		}

		// Phase 3: max hold enforcement
		if cfg.ForceCloseOnHold && cfg.MaxHoldHours > 0 {
			for _, row := range grp.Rows {
				sym := row.Symbol
				inv := inventory[sym]
				opened, hasOpen := openTime[sym]
				if inv <= 0 || !hasOpen {
					continue
				}
				heldHours := float64(ts-opened) / 3600.0
				if heldHours < float64(cfg.MaxHoldHours) {
					continue
				}
				c := row.Close
				proceeds := inv * c * (1 - cfg.MakerFee)
				cash += proceeds
				realized := (c - costBasis[sym]) * inv
				inventory[sym] = 0
				costBasis[sym] = 0
				delete(openTime, sym)
				trades = append(trades, TradeRecord{
					Timestamp: ts, Symbol: sym, Side: "sell",
					Price: c, Quantity: inv, Notional: inv * c,
					Fee: inv * c * cfg.MakerFee,
					CashAfter: cash, InventoryAfter: 0, RealizedPnl: realized,
					Reason: "max_hold",
				})
			}
		}

		// Equity snapshot
		invValue := 0.0
		for _, row := range grp.Rows {
			invValue += inventory[row.Symbol] * row.Close
		}
		equityCurve = append(equityCurve, cash+invValue)
	}

	metrics := computeMetrics(equityCurve)
	return SharedCashResult{
		EquityCurve: equityCurve,
		Trades:      trades,
		Metrics:     metrics,
	}
}

func computeMetrics(eq []float64) map[string]float64 {
	m := map[string]float64{
		"total_return":      0,
		"sortino":           0,
		"mean_hourly_return": 0,
	}
	if len(eq) < 2 {
		return m
	}
	m["total_return"] = eq[len(eq)-1]/eq[0] - 1
	m["sortino"] = Sortino(eq, HourlyPeriodsPerYear)
	m["max_drawdown"] = MaxDrawdown(eq)
	returns := StepReturns(eq)
	sum := 0.0
	for _, r := range returns {
		sum += r
	}
	m["mean_hourly_return"] = sum / float64(len(returns))
	return m
}

func applySharedLag(bars []SymbolBar, actions []SymbolAction, lag int) []SymbolAction {
	// Group actions by symbol, sort by timestamp
	bySymbol := make(map[string][]SymbolAction)
	for _, a := range actions {
		bySymbol[a.Symbol] = append(bySymbol[a.Symbol], a)
	}
	// Group bar timestamps by symbol
	barTS := make(map[string][]int64)
	for _, b := range bars {
		barTS[b.Symbol] = append(barTS[b.Symbol], b.Timestamp)
	}

	var result []SymbolAction
	for sym, symActs := range bySymbol {
		sort.Slice(symActs, func(i, j int) bool {
			return symActs[i].Timestamp < symActs[j].Timestamp
		})
		ts := barTS[sym]
		sort.Slice(ts, func(i, j int) bool { return ts[i] < ts[j] })
		usable := len(symActs)
		if usable > len(ts) {
			usable = len(ts)
		}
		if usable <= lag {
			continue
		}
		for i := 0; i < usable-lag; i++ {
			a := symActs[i]
			a.Timestamp = ts[i+lag]
			result = append(result, a)
		}
	}
	return result
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func quantizeDown(v, step float64) float64 {
	if step <= 0 {
		return v
	}
	return math.Floor(v/step) * step
}

func quantizeUp(v, step float64) float64 {
	if step <= 0 {
		return v
	}
	return math.Ceil(v/step) * step
}

func resolveScale(amount float64) float64 {
	if amount > 1.5 {
		return 100.0
	}
	return 1.0
}
