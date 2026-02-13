package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"time"

	"github.com/alpacahq/alpaca-trade-api-go/v3/alpaca"
	"github.com/shopspring/decimal"
)

// RampConfig controls how we gradually accumulate a position.
type RampConfig struct {
	Symbol       string
	Side         alpaca.Side
	TargetQty    float64       // Total qty to accumulate
	DurationMins float64       // Ramp window (default 60 min)
	PollInterval time.Duration // How often to check/reorder (default 30s)
	PriceOffset  float64       // Initial offset from market (0.0004 = 4bps)
}

// RampIntoPosition gradually accumulates a position over time,
// linearly moving the limit price from passive to aggressive.
// Mirrors the Python ramp_into_position pattern.
func (t *Trader) RampIntoPosition(ctx context.Context, cfg RampConfig) error {
	if cfg.DurationMins <= 0 {
		cfg.DurationMins = 60
	}
	if cfg.PollInterval <= 0 {
		cfg.PollInterval = 30 * time.Second
	}
	if cfg.PriceOffset <= 0 {
		cfg.PriceOffset = 0.0004
	}

	sym := remapSymbol(cfg.Symbol)
	startTime := time.Now()
	endTime := startTime.Add(time.Duration(cfg.DurationMins) * time.Minute)

	log.Printf("[ramp] Starting %s %s target_qty=%.6f over %.0f min",
		cfg.Side, sym, cfg.TargetQty, cfg.DurationMins)

	for {
		select {
		case <-ctx.Done():
			log.Printf("[ramp] %s cancelled", sym)
			return ctx.Err()
		default:
		}

		// Check current position
		currentQty := t.currentPositionQty(sym, cfg.Side)
		remaining := cfg.TargetQty - currentQty
		if remaining <= 0 {
			log.Printf("[ramp] %s target reached (%.6f >= %.6f)", sym, currentQty, cfg.TargetQty)
			return nil
		}

		// Check time
		now := time.Now()
		if now.After(endTime) {
			log.Printf("[ramp] %s time expired, filled %.6f / %.6f", sym, currentQty, cfg.TargetQty)
			return fmt.Errorf("ramp expired: filled %.6f / %.6f", currentQty, cfg.TargetQty)
		}

		// Calculate progress (0.0 = start, 1.0 = end)
		elapsed := now.Sub(startTime).Minutes()
		progress := math.Min(elapsed/cfg.DurationMins, 1.0)

		// Get current quote
		quote, err := t.GetLatestQuote(cfg.Symbol)
		if err != nil {
			log.Printf("[ramp] %s quote error: %v, retrying", sym, err)
			time.Sleep(cfg.PollInterval)
			continue
		}

		// Calculate ramp price: start passive, linearly move to aggressive
		price := rampPrice(cfg.Side, quote.BidPrice, quote.AskPrice, progress, cfg.PriceOffset)

		log.Printf("[ramp] %s progress=%.1f%% qty_remaining=%.6f price=%.4f bid=%.4f ask=%.4f",
			sym, progress*100, remaining, price, quote.BidPrice, quote.AskPrice)

		// Cancel existing orders for this symbol/side
		t.cancelOrdersForSymbol(sym, cfg.Side)

		// Place limit order for remaining qty
		_, err = t.PlaceLimitOrder(cfg.Symbol, remaining, cfg.Side, price)
		if err != nil {
			log.Printf("[ramp] %s order error: %v", sym, err)
		}

		time.Sleep(cfg.PollInterval)
	}
}

// rampPrice calculates the limit price based on ramp progress.
// At progress=0: passive side (below bid for buys, above ask for sells)
// At progress=1: aggressive side (at ask for buys, at bid for sells)
func rampPrice(side alpaca.Side, bid, ask, progress, offset float64) float64 {
	if side == alpaca.Buy {
		startPrice := bid * (1 - offset) // Below bid
		endPrice := ask                  // At ask (cross spread)
		return startPrice + (endPrice-startPrice)*progress
	}
	// Sell
	startPrice := ask * (1 + offset) // Above ask
	endPrice := bid                  // At bid (cross spread)
	return startPrice + (endPrice-startPrice)*progress
}

// currentPositionQty returns the absolute qty held for a symbol on the given side.
func (t *Trader) currentPositionQty(symbol string, side alpaca.Side) float64 {
	pos, err := t.client.GetPosition(symbol)
	if err != nil {
		return 0
	}
	qty, exact := pos.Qty.Float64()
	if !exact {
		log.Printf("[ramp] qty conversion inexact for %s", symbol)
	}
	qty = math.Abs(qty)

	// Check side matches
	if side == alpaca.Buy && pos.Side == "short" {
		return 0
	}
	if side == alpaca.Sell && pos.Side == "long" {
		return 0
	}
	return qty
}

// cancelOrdersForSymbol cancels all open orders for a symbol on a given side.
func (t *Trader) cancelOrdersForSymbol(symbol string, side alpaca.Side) {
	orders, err := t.GetOrders("open", 100)
	if err != nil {
		return
	}
	for _, o := range orders {
		if o.Symbol == symbol && o.Side == side {
			_ = t.CancelOrder(o.ID)
		}
	}
}

// ---------- Notional-based ramp (allocate by $ amount) ----------

// RampByNotional calculates target qty from a dollar amount and current price,
// then calls RampIntoPosition.
func (t *Trader) RampByNotional(ctx context.Context, symbol string, side alpaca.Side,
	notionalUSD float64, durationMins float64) error {

	quote, err := t.GetLatestQuote(symbol)
	if err != nil {
		return fmt.Errorf("quote for notional calc: %w", err)
	}

	if quote.MidPrice <= 0 {
		return fmt.Errorf("invalid mid price %.4f for %s", quote.MidPrice, symbol)
	}

	targetQty := notionalUSD / quote.MidPrice

	// Enforce minimum notional
	minNotional := 1.0
	if isCrypto(symbol) {
		minNotional = 10.0
	}
	if notionalUSD < minNotional {
		return fmt.Errorf("notional $%.2f below minimum $%.2f for %s", notionalUSD, minNotional, symbol)
	}

	return t.RampIntoPosition(ctx, RampConfig{
		Symbol:       symbol,
		Side:         side,
		TargetQty:    targetQty,
		DurationMins: durationMins,
	})
}

// RampByAllocationPct calculates target qty from a % of account equity.
func (t *Trader) RampByAllocationPct(ctx context.Context, symbol string, side alpaca.Side,
	pct float64, durationMins float64) error {

	acct, err := t.GetAccount()
	if err != nil {
		return fmt.Errorf("account for allocation: %w", err)
	}

	equity, _ := acct.Equity.Float64()
	if equity <= 0 {
		return fmt.Errorf("invalid equity: %v", equity)
	}
	notional := equity * pct / 100.0

	log.Printf("[ramp] %s %.1f%% of $%.2f equity = $%.2f notional",
		symbol, pct, equity, notional)

	return t.RampByNotional(ctx, symbol, side, notional, durationMins)
}

// ---------- Leverage-aware ordering ----------

// MaxLeverageOrder places an order but caps it to stay within a leverage limit.
func (t *Trader) MaxLeverageOrder(symbol string, qty float64, side alpaca.Side,
	price float64, maxLeverage float64) (*alpaca.Order, error) {

	acct, err := t.GetAccount()
	if err != nil {
		return nil, err
	}

	equity, _ := acct.Equity.Float64()
	if equity <= 0 {
		return nil, fmt.Errorf("invalid equity: %v", equity)
	}
	maxExposure := equity * maxLeverage

	// Sum current exposure
	positions, _ := t.GetPositions()
	currentExposure := 0.0
	for _, p := range positions {
		mv := derefDec(p.MarketValue)
		f, _ := mv.Abs().Float64()
		currentExposure += f
	}

	orderNotional := qty * price
	available := maxExposure - currentExposure

	if available <= 0 {
		return nil, fmt.Errorf("at max leverage (exposure $%.0f / max $%.0f)", currentExposure, maxExposure)
	}

	if orderNotional > available {
		if price <= 0 {
			return nil, fmt.Errorf("invalid price %.4f for leverage calc", price)
		}
		qty = available / price
		log.Printf("[leverage] Capped order to qty=%.6f (available $%.0f)", qty, available)
	}

	qtyDec := decimal.NewFromFloat(qty)
	priceDec := decimal.NewFromFloat(price)
	sym := remapSymbol(symbol)
	tif := timeInForce(sym, qtyDec)

	return t.client.PlaceOrder(alpaca.PlaceOrderRequest{
		Symbol:      sym,
		Qty:         &qtyDec,
		Side:        side,
		Type:        alpaca.Limit,
		TimeInForce: tif,
		LimitPrice:  &priceDec,
	})
}
