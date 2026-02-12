package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/alpacahq/alpaca-trade-api-go/v3/alpaca"
	"github.com/shopspring/decimal"
)

func usage() {
	fmt.Fprintf(os.Stderr, `alpaca_go - Alpaca trading CLI (Go)

Usage:
  alpaca_go <command> [args]

Commands:
  status                          Account status, positions, and open orders
  positions                       List all open positions
  orders [status]                 List orders (default: open)
  buy <symbol> <qty> [price]      Buy (market if no price, limit if price given)
  sell <symbol> <qty> [price]     Sell (market if no price, limit if price given)
  close <symbol>                  Close position for symbol
  close-all                       Close all positions and cancel all orders
  cancel <order-id>               Cancel a specific order
  cancel-all                      Cancel all open orders
  asset <symbol>                  Check if asset is tradable
  clock                           Market clock status
  quote <symbol>                  Get latest bid/ask quote
  ramp <symbol> <side> <qty> [mins]  Ramp into position over time (default 60 min)
  ramp-pct <symbol> <side> <pct> [mins]  Ramp by %% of equity
  watch <symbol> <side> <qty> <price> [tolerance%%] [expiry-mins]
                                  Watch for price trigger, then enter position
  watchers                        List active watchers
`)
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}

	cfg, err := LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Config error: %v\n", err)
		os.Exit(1)
	}

	trader := NewTrader(cfg)
	cmd := strings.ToLower(os.Args[1])

	switch cmd {
	case "status":
		cmdStatus(trader)
	case "positions", "pos":
		cmdPositions(trader)
	case "orders":
		status := "open"
		if len(os.Args) > 2 {
			status = os.Args[2]
		}
		cmdOrders(trader, status)
	case "buy":
		if len(os.Args) < 4 {
			fmt.Fprintln(os.Stderr, "Usage: alpaca_go buy <symbol> <qty> [price]")
			os.Exit(1)
		}
		cmdTrade(trader, os.Args[2], os.Args[3], alpaca.Buy, optArg(4))
	case "sell":
		if len(os.Args) < 4 {
			fmt.Fprintln(os.Stderr, "Usage: alpaca_go sell <symbol> <qty> [price]")
			os.Exit(1)
		}
		cmdTrade(trader, os.Args[2], os.Args[3], alpaca.Sell, optArg(4))
	case "close":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: alpaca_go close <symbol>")
			os.Exit(1)
		}
		cmdClose(trader, os.Args[2])
	case "close-all":
		cmdCloseAll(trader)
	case "cancel":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: alpaca_go cancel <order-id>")
			os.Exit(1)
		}
		cmdCancel(trader, os.Args[2])
	case "cancel-all":
		cmdCancelAll(trader)
	case "asset":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: alpaca_go asset <symbol>")
			os.Exit(1)
		}
		cmdAsset(trader, os.Args[2])
	case "clock":
		cmdClock(trader)
	case "quote":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: alpaca_go quote <symbol>")
			os.Exit(1)
		}
		cmdQuote(trader, os.Args[2])
	case "ramp":
		if len(os.Args) < 5 {
			fmt.Fprintln(os.Stderr, "Usage: alpaca_go ramp <symbol> <buy|sell> <qty> [duration-mins]")
			os.Exit(1)
		}
		cmdRamp(trader, os.Args[2], os.Args[3], os.Args[4], optArg(5))
	case "ramp-pct":
		if len(os.Args) < 5 {
			fmt.Fprintln(os.Stderr, "Usage: alpaca_go ramp-pct <symbol> <buy|sell> <pct> [duration-mins]")
			os.Exit(1)
		}
		cmdRampPct(trader, os.Args[2], os.Args[3], os.Args[4], optArg(5))
	case "watch":
		if len(os.Args) < 6 {
			fmt.Fprintln(os.Stderr, "Usage: alpaca_go watch <symbol> <buy|sell> <qty> <price> [tolerance%] [expiry-mins]")
			os.Exit(1)
		}
		cmdWatch(trader, os.Args[2:])
	case "watchers":
		wm := NewWatcherManager(trader)
		wm.PrintWatcherStatus()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", cmd)
		usage()
	}
}

func optArg(idx int) string {
	if idx < len(os.Args) {
		return os.Args[idx]
	}
	return ""
}

// ---------- Command implementations ----------

func cmdStatus(t *Trader) {
	mode := "PAPER"
	if !t.cfg.Paper {
		mode = "LIVE"
	}
	fmt.Printf("=== Alpaca Account Status [%s] ===\n\n", mode)

	acct, err := t.GetAccount()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting account: %v\n", err)
		os.Exit(1)
	}

	balChange := acct.Equity.Sub(acct.LastEquity)
	balPct := decimal.Zero
	if !acct.LastEquity.IsZero() {
		balPct = balChange.Div(acct.LastEquity).Mul(decimal.NewFromInt(100))
	}

	fmt.Printf("  Equity:          %s\n", formatMoney(acct.Equity))
	fmt.Printf("  Cash:            %s\n", formatMoney(acct.Cash))
	fmt.Printf("  Buying Power:    %s\n", formatMoney(acct.BuyingPower))
	fmt.Printf("  Day P&L:         %s (%s%%)\n", formatMoney(balChange), balPct.StringFixed(2))
	fmt.Printf("  Long MV:         %s\n", formatMoney(acct.LongMarketValue))
	fmt.Printf("  Short MV:        %s\n", formatMoney(acct.ShortMarketValue))
	fmt.Printf("  Multiplier:      %s\n", acct.Multiplier.String())
	fmt.Printf("  PDT:             %v\n", acct.PatternDayTrader)
	fmt.Printf("  Daytrading BP:   %s\n", formatMoney(acct.DaytradingBuyingPower))
	fmt.Println()

	// Clock
	clock, err := t.GetClock()
	if err == nil {
		status := "CLOSED"
		if clock.IsOpen {
			status = "OPEN"
		}
		fmt.Printf("  Market:          %s  (next open: %s, next close: %s)\n",
			status,
			clock.NextOpen.Format("2006-01-02 15:04 MST"),
			clock.NextClose.Format("2006-01-02 15:04 MST"),
		)
	}
	fmt.Println()

	// Positions
	positions, err := t.GetPositions()
	if err != nil {
		fmt.Fprintf(os.Stderr, "  Error fetching positions: %v\n", err)
	} else if len(positions) == 0 {
		fmt.Println("  Positions: (none)")
	} else {
		fmt.Printf("  Positions (%d):\n", len(positions))
		fmt.Printf("    %-12s %8s %12s %12s %12s %10s\n",
			"Symbol", "Qty", "Mkt Value", "Avg Price", "P&L", "P&L %")
		fmt.Printf("    %s\n", strings.Repeat("-", 70))
		for _, p := range positions {
			plPct := derefDec(p.UnrealizedPLPC).Mul(decimal.NewFromInt(100))
			fmt.Printf("    %-12s %8s %12s %12s %12s %9s%%\n",
				unremapSymbol(p.Symbol),
				p.Qty.String(),
				formatMoney(derefDec(p.MarketValue)),
				formatMoney(p.AvgEntryPrice),
				formatMoney(derefDec(p.UnrealizedPL)),
				plPct.StringFixed(2),
			)
		}
	}
	fmt.Println()

	// Open orders
	orders, err := t.GetOrders("open", 50)
	if err != nil {
		fmt.Fprintf(os.Stderr, "  Error fetching orders: %v\n", err)
	} else if len(orders) == 0 {
		fmt.Println("  Open Orders: (none)")
	} else {
		fmt.Printf("  Open Orders (%d):\n", len(orders))
		fmt.Printf("    %-12s %6s %8s %10s %12s %8s\n",
			"Symbol", "Side", "Qty", "Type", "Limit", "Status")
		fmt.Printf("    %s\n", strings.Repeat("-", 60))
		for _, o := range orders {
			limitStr := "-"
			if o.LimitPrice != nil {
				limitStr = formatMoney(*o.LimitPrice)
			}
			fmt.Printf("    %-12s %6s %8s %10s %12s %8s\n",
				unremapSymbol(o.Symbol),
				o.Side,
				derefDec(o.Qty).String(),
				o.Type,
				limitStr,
				o.Status,
			)
		}
	}
}

func cmdPositions(t *Trader) {
	positions, err := t.GetPositions()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if len(positions) == 0 {
		fmt.Println("No open positions.")
		return
	}
	fmt.Printf("%-12s %8s %12s %12s %12s %10s\n",
		"Symbol", "Qty", "Mkt Value", "Avg Price", "P&L", "P&L %")
	fmt.Println(strings.Repeat("-", 70))
	for _, p := range positions {
		plPct := derefDec(p.UnrealizedPLPC).Mul(decimal.NewFromInt(100))
		fmt.Printf("%-12s %8s %12s %12s %12s %9s%%\n",
			unremapSymbol(p.Symbol),
			p.Qty.String(),
			formatMoney(derefDec(p.MarketValue)),
			formatMoney(p.AvgEntryPrice),
			formatMoney(derefDec(p.UnrealizedPL)),
			plPct.StringFixed(2),
		)
	}
}

func cmdOrders(t *Trader, status string) {
	orders, err := t.GetOrders(status, 100)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if len(orders) == 0 {
		fmt.Printf("No %s orders.\n", status)
		return
	}
	fmt.Printf("%-36s %-12s %6s %8s %10s %12s %12s %8s\n",
		"Order ID", "Symbol", "Side", "Qty", "Type", "Limit", "Stop", "Status")
	fmt.Println(strings.Repeat("-", 108))
	for _, o := range orders {
		limitStr := "-"
		if o.LimitPrice != nil {
			limitStr = formatMoney(*o.LimitPrice)
		}
		stopStr := "-"
		if o.StopPrice != nil {
			stopStr = formatMoney(*o.StopPrice)
		}
		fmt.Printf("%-36s %-12s %6s %8s %10s %12s %12s %8s\n",
			o.ID,
			unremapSymbol(o.Symbol),
			o.Side,
			derefDec(o.Qty).String(),
			o.Type,
			limitStr,
			stopStr,
			o.Status,
		)
	}
}

func cmdTrade(t *Trader, symbol, qtyStr string, side alpaca.Side, priceStr string) {
	qty, err := strconv.ParseFloat(qtyStr, 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid qty: %s\n", qtyStr)
		os.Exit(1)
	}

	var order *alpaca.Order
	if priceStr == "" {
		// Market order
		order, err = t.PlaceMarketOrder(symbol, qty, side)
	} else {
		// Limit order
		price, perr := strconv.ParseFloat(priceStr, 64)
		if perr != nil {
			fmt.Fprintf(os.Stderr, "Invalid price: %s\n", priceStr)
			os.Exit(1)
		}
		order, err = t.PlaceLimitOrder(symbol, qty, side, price)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Order failed: %v\n", err)
		os.Exit(1)
	}

	limitStr := "market"
	if order.LimitPrice != nil {
		limitStr = formatMoney(*order.LimitPrice)
	}
	fmt.Printf("Order placed: %s %s %s x%s @ %s [%s]\n",
		order.Side, order.Symbol, order.Type, derefDec(order.Qty).String(), limitStr, order.ID)
}

func cmdClose(t *Trader, symbol string) {
	order, err := t.ClosePosition(symbol)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Close failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Closing %s: %s x%s [%s]\n",
		unremapSymbol(order.Symbol), order.Side, derefDec(order.Qty).String(), order.ID)
}

func cmdCloseAll(t *Trader) {
	orders, err := t.CloseAllPositions()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Close all failed: %v\n", err)
		os.Exit(1)
	}
	if len(orders) == 0 {
		fmt.Println("No positions to close.")
		return
	}
	for _, o := range orders {
		fmt.Printf("Closing %s: %s x%s [%s]\n",
			unremapSymbol(o.Symbol), o.Side, derefDec(o.Qty).String(), o.ID)
	}
}

func cmdCancel(t *Trader, orderID string) {
	err := t.CancelOrder(orderID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Cancel failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Cancelled order %s\n", orderID)
}

func cmdCancelAll(t *Trader) {
	err := t.CancelAllOrders()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Cancel all failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("All orders cancelled.")
}

func cmdAsset(t *Trader, symbol string) {
	asset, err := t.GetAsset(symbol)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Asset not found: %v\n", err)
		os.Exit(1)
	}
	tradable := "NO"
	if asset.Tradable {
		tradable = "YES"
	}
	shortable := "NO"
	if asset.Shortable {
		shortable = "YES"
	}
	fractionable := "NO"
	if asset.Fractionable {
		fractionable = "YES"
	}
	fmt.Printf("Symbol:       %s\n", asset.Symbol)
	fmt.Printf("Name:         %s\n", asset.Name)
	fmt.Printf("Class:        %s\n", asset.Class)
	fmt.Printf("Exchange:     %s\n", asset.Exchange)
	fmt.Printf("Tradable:     %s\n", tradable)
	fmt.Printf("Shortable:    %s\n", shortable)
	fmt.Printf("Fractionable: %s\n", fractionable)
	fmt.Printf("Status:       %s\n", asset.Status)
}

func cmdClock(t *Trader) {
	clock, err := t.GetClock()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	status := "CLOSED"
	if clock.IsOpen {
		status = "OPEN"
	}
	fmt.Printf("Market:     %s\n", status)
	fmt.Printf("Timestamp:  %s\n", clock.Timestamp.Format("2006-01-02 15:04:05 MST"))
	fmt.Printf("Next Open:  %s\n", clock.NextOpen.Format("2006-01-02 15:04:05 MST"))
	fmt.Printf("Next Close: %s\n", clock.NextClose.Format("2006-01-02 15:04:05 MST"))
}

func cmdQuote(t *Trader, symbol string) {
	q, err := t.GetLatestQuote(symbol)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Quote error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Symbol:  %s\n", q.Symbol)
	fmt.Printf("Bid:     $%.6f\n", q.BidPrice)
	fmt.Printf("Ask:     $%.6f\n", q.AskPrice)
	fmt.Printf("Mid:     $%.6f\n", q.MidPrice)
	fmt.Printf("Spread:  %.4f%%\n", q.SpreadPct()*100)
}

func parseSide(s string) alpaca.Side {
	switch strings.ToLower(s) {
	case "sell", "short":
		return alpaca.Sell
	default:
		return alpaca.Buy
	}
}

func cmdRamp(t *Trader, symbol, sideStr, qtyStr, minsStr string) {
	side := parseSide(sideStr)
	qty, err := strconv.ParseFloat(qtyStr, 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid qty: %s\n", qtyStr)
		os.Exit(1)
	}
	mins := 60.0
	if minsStr != "" {
		mins, _ = strconv.ParseFloat(minsStr, 64)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	fmt.Printf("Ramping %s %s x%.6f over %.0f min (Ctrl+C to cancel)\n", side, symbol, qty, mins)
	err = t.RampIntoPosition(ctx, RampConfig{
		Symbol:       symbol,
		Side:         side,
		TargetQty:    qty,
		DurationMins: mins,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Ramp error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Ramp complete.")
}

func cmdRampPct(t *Trader, symbol, sideStr, pctStr, minsStr string) {
	side := parseSide(sideStr)
	pct, err := strconv.ParseFloat(pctStr, 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Invalid pct: %s\n", pctStr)
		os.Exit(1)
	}
	mins := 60.0
	if minsStr != "" {
		mins, _ = strconv.ParseFloat(minsStr, 64)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	fmt.Printf("Ramping %s %s %.1f%% of equity over %.0f min (Ctrl+C to cancel)\n",
		side, symbol, pct, mins)
	err = t.RampByAllocationPct(ctx, symbol, side, pct, mins)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Ramp error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Ramp complete.")
}

func cmdWatch(t *Trader, args []string) {
	symbol := args[0]
	side := args[1]
	qty, _ := strconv.ParseFloat(args[2], 64)
	price, _ := strconv.ParseFloat(args[3], 64)
	tolerance := 0.005 // 0.5% default
	if len(args) > 4 {
		tolerance, _ = strconv.ParseFloat(args[4], 64)
		tolerance /= 100.0 // Convert from % to decimal
	}
	expiryMins := 120.0 // 2h default
	if len(args) > 5 {
		expiryMins, _ = strconv.ParseFloat(args[5], 64)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	wm := NewWatcherManager(t)
	err := wm.StartWatcher(WatcherConfig{
		Symbol:       symbol,
		Side:         side,
		TargetQty:    qty,
		LimitPrice:   price,
		TolerancePct: tolerance,
		ExpiresAt:    time.Now().Add(time.Duration(expiryMins) * time.Minute),
		PollInterval: 30 * time.Second,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Watch error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Watching %s %s x%.6f @ $%.4f (tolerance %.2f%%, expires in %.0f min)\n",
		side, symbol, qty, price, tolerance*100, expiryMins)
	fmt.Println("Press Ctrl+C to cancel...")

	wm.WaitForAllWatchers(ctx)
	wm.PrintWatcherStatus()
}
