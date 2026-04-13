package trade

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

// TradingLoop runs the hourly trading cycle.
type TradingLoop struct {
	Client    *BinanceClient
	Guard     *DeathSpiralGuard
	Singleton *SingletonLock
	Symbols   []string
	Interval  time.Duration
}

// NewTradingLoop creates a trading loop with safety guards.
func NewTradingLoop(stateDir string, symbols []string) (*TradingLoop, error) {
	lock, err := AcquireSingleton(stateDir)
	if err != nil {
		return nil, err
	}

	return &TradingLoop{
		Client:    NewBinanceClient(),
		Guard:     NewDeathSpiralGuard(stateDir),
		Singleton: lock,
		Symbols:   symbols,
		Interval:  1 * time.Hour,
	}, nil
}

// Run starts the trading loop. Blocks until interrupted.
func (tl *TradingLoop) Run() {
	defer tl.Singleton.Release()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	log.Printf("trading loop started: %d symbols, interval=%s", len(tl.Symbols), tl.Interval)

	// Run immediately on start
	tl.tick()

	ticker := time.NewTicker(tl.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			tl.tick()
		case sig := <-sigCh:
			log.Printf("received %s, shutting down", sig)
			return
		}
	}
}

func (tl *TradingLoop) tick() {
	log.Printf("tick: checking %d symbols", len(tl.Symbols))

	for _, sym := range tl.Symbols {
		quote, err := tl.Client.GetQuote(sym)
		if err != nil {
			log.Printf("  %s: quote error: %v", sym, err)
			continue
		}
		log.Printf("  %s: bid=%.2f ask=%.2f last=%.2f", sym, quote.Bid, quote.Ask, quote.Last)

		// TODO: integrate ONNX inference here
		// 1. Load latest bars + forecast cache
		// 2. Run ONNX model inference
		// 3. Decode actions
		// 4. Execute via client with death spiral guard
	}
}

// ExecuteAction places an order with safety checks.
func (tl *TradingLoop) ExecuteAction(symbol string, side string, price, qty float64) error {
	if side == "SELL" {
		if err := tl.Guard.CheckSell(symbol, price); err != nil {
			return err
		}
	}

	resp, err := tl.Client.PlaceOrder(OrderRequest{
		Symbol:   symbol,
		Side:     side,
		Type:     "LIMIT",
		Quantity: qty,
		Price:    price,
	})
	if err != nil {
		return fmt.Errorf("place %s order: %w", side, err)
	}

	log.Printf("  %s %s: order_id=%s qty=%.6f @ %.2f",
		side, symbol, resp.OrderID, qty, price)

	if side == "BUY" {
		tl.Guard.RecordBuy(symbol, price)
	}
	return nil
}
