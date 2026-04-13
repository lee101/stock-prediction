package sim

import "math"

// Simulate runs a single-symbol trading simulation matching market_sim.c exactly.
// bars and actions must be the same length. Decision lag should be applied before calling.
func Simulate(bars []Bar, actions []Action, cfg SimConfig) SimResult {
	n := len(bars)
	if n == 0 || len(actions) < n {
		return SimResult{
			FinalEquity: cfg.InitialCash,
			EquityCurve: []float64{cfg.InitialCash},
		}
	}

	cash := cfg.InitialCash
	inventory := 0.0
	marginCostTotal := 0.0
	barsInPosition := 0
	numTrades := 0
	iscale := cfg.IntensityScale
	fillBuf := cfg.FillBufferPct

	eq := make([]float64, n+1)
	eq[0] = cash

	for i := 0; i < n; i++ {
		c := bars[i].Close
		h := bars[i].High
		l := bars[i].Low
		bp := actions[i].BuyPrice
		sp := actions[i].SellPrice
		ba := actions[i].BuyAmount * iscale
		sa := actions[i].SellAmount * iscale
		if ba > 100 {
			ba = 100
		}
		if sa > 100 {
			sa = 100
		}
		ba /= 100
		sa /= 100

		equity := cash + inventory*c

		// margin interest
		if cash < 0 {
			interest := (-cash) * cfg.MarginHourlyRate
			cash -= interest
			marginCostTotal += interest
		}
		if inventory < 0 {
			bv := (-inventory) * c
			interest := bv * cfg.MarginHourlyRate
			cash -= interest
			marginCostTotal += interest
		}

		// force close
		if cfg.MaxHoldBars > 0 && inventory > 0 && barsInPosition >= cfg.MaxHoldBars {
			fp := c * 0.999
			cash += inventory * fp * (1 - cfg.MakerFee)
			numTrades++
			inventory = 0
			barsInPosition = 0
			eq[i+1] = cash
			continue
		}

		// min edge
		edge := 0.0
		if bp > 0 && sp > 0 {
			edge = (sp - bp) / bp
		}
		if cfg.MinEdge > 0 && edge < cfg.MinEdge {
			if inventory > 0 {
				barsInPosition++
			}
			eq[i+1] = cash + inventory*c
			continue
		}

		// sell first
		soldThisBar := false
		if sa > 0 && sp > 0 && h >= sp*(1+fillBuf) {
			sellQty := 0.0
			if inventory > 0 {
				sellQty = sa * inventory
				if sellQty > inventory {
					sellQty = inventory
				}
			} else if cfg.CanShort {
				eqPos := equity
				if eqPos < 0 {
					eqPos = 0
				}
				maxSV := cfg.MaxLeverage * eqPos
				q1 := sa * maxSV / (sp * (1 + cfg.MakerFee))
				q2 := maxSV / (sp * (1 + cfg.MakerFee))
				sellQty = math.Min(q1, q2)
			}
			if sellQty < 0 {
				sellQty = 0
			}
			if sellQty > 0 {
				cash += sellQty * sp * (1 - cfg.MakerFee)
				inventory -= sellQty
				numTrades++
				soldThisBar = true
				if inventory <= 0 {
					barsInPosition = 0
				}
			}
		}

		// buy (no same-bar roundtrip)
		if !soldThisBar && ba > 0 && bp > 0 && l <= bp*(1-fillBuf) {
			eqPos := equity
			if eqPos < 0 {
				eqPos = 0
			}
			maxBV := cfg.MaxLeverage*eqPos - inventory*bp
			if maxBV > 0 {
				buyQty := ba * maxBV / (bp * (1 + cfg.MakerFee))
				if buyQty > 0 {
					cash -= buyQty * bp * (1 + cfg.MakerFee)
					inventory += buyQty
					numTrades++
				}
			}
		}

		if inventory > 0 {
			barsInPosition++
		} else {
			barsInPosition = 0
		}

		eq[i+1] = cash + inventory*c
	}

	// close remaining position
	if n > 0 && inventory != 0 {
		lc := bars[n-1].Close
		if inventory > 0 {
			cash += inventory * lc * (1 - cfg.MakerFee)
		} else {
			cash -= (-inventory) * lc * (1 + cfg.MakerFee)
		}
		inventory = 0
		eq[n] = cash
	}

	totalReturn := 0.0
	if eq[0] > 0 {
		totalReturn = eq[n]/eq[0] - 1
	}

	return SimResult{
		TotalReturn:     totalReturn,
		Sortino:         Sortino(eq, HourlyPeriodsPerYear),
		MaxDrawdown:     MaxDrawdown(eq),
		FinalEquity:     eq[n],
		NumTrades:       numTrades,
		MarginCostTotal: marginCostTotal,
		EquityCurve:     eq,
	}
}

// SimulateBatch runs N simulations with different configs on the same data.
func SimulateBatch(bars []Bar, actions []Action, configs []SimConfig) []SimResult {
	results := make([]SimResult, len(configs))
	for i, cfg := range configs {
		results[i] = Simulate(bars, actions, cfg)
	}
	return results
}

// ApplyDecisionLag shifts actions forward by lag bars.
// Returns shortened bars and actions slices where action[i] was generated from bar[i-lag].
func ApplyDecisionLag(bars []Bar, actions []Action, lag int) ([]Bar, []Action) {
	if lag <= 0 {
		return bars, actions
	}
	n := len(actions)
	if lag >= n || lag >= len(bars) {
		return nil, nil
	}
	usable := n - lag
	if usable > len(bars)-lag {
		usable = len(bars) - lag
	}
	return bars[lag : lag+usable], actions[:usable]
}
