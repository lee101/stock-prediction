package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/alpacahq/alpaca-trade-api-go/v3/alpaca"
)

// WatcherState tracks the lifecycle of a position watcher.
type WatcherState string

const (
	WatcherInitializing   WatcherState = "initializing"
	WatcherWaiting        WatcherState = "waiting_for_trigger"
	WatcherSubmitting     WatcherState = "submitting_order"
	WatcherAwaitingFill   WatcherState = "awaiting_fill"
	WatcherPositionOpen   WatcherState = "position_open"
	WatcherExiting        WatcherState = "exiting"
	WatcherDone           WatcherState = "done"
	WatcherExpired        WatcherState = "expired"
	WatcherError          WatcherState = "error"
)

// WatcherConfig defines a position watcher.
type WatcherConfig struct {
	Symbol        string        `json:"symbol"`
	Side          string        `json:"side"`    // "buy" or "sell"
	TargetQty     float64       `json:"target_qty"`
	LimitPrice    float64       `json:"limit_price"`    // Target entry price
	TolerancePct  float64       `json:"tolerance_pct"`  // Price tolerance band (e.g. 0.005 = 0.5%)
	ExpiresAt     time.Time     `json:"expires_at"`
	PollInterval  time.Duration `json:"poll_interval"`
	RampOnTrigger bool          `json:"ramp_on_trigger"` // Use ramp instead of single order
	RampMins      float64       `json:"ramp_mins"`       // Ramp duration if ramp_on_trigger
}

// WatcherStatus is the persisted state for a watcher.
type WatcherStatus struct {
	Config      WatcherConfig `json:"config"`
	State       WatcherState  `json:"state"`
	PositionQty float64       `json:"position_qty"`
	OrderQty    float64       `json:"order_qty"`
	StartedAt   time.Time     `json:"started_at"`
	UpdatedAt   time.Time     `json:"updated_at"`
	Error       string        `json:"error,omitempty"`
}

// WatcherManager runs multiple position watchers as goroutines.
type WatcherManager struct {
	trader   *Trader
	watchers map[string]context.CancelFunc // symbol -> cancel
	mu       sync.Mutex
	stateDir string
}

func NewWatcherManager(trader *Trader) *WatcherManager {
	stateDir := filepath.Join(".", ".watcher_state")
	os.MkdirAll(stateDir, 0755)
	return &WatcherManager{
		trader:   trader,
		watchers: make(map[string]context.CancelFunc),
		stateDir: stateDir,
	}
}

// StartWatcher launches a goroutine to watch for an entry point and accumulate a position.
func (wm *WatcherManager) StartWatcher(cfg WatcherConfig) error {
	wm.mu.Lock()
	defer wm.mu.Unlock()

	key := cfg.Symbol + ":" + cfg.Side
	if cancel, exists := wm.watchers[key]; exists {
		cancel() // Stop existing watcher for same symbol/side
	}

	if cfg.PollInterval <= 0 {
		cfg.PollInterval = 30 * time.Second
	}

	ctx, cancel := context.WithCancel(context.Background())
	wm.watchers[key] = cancel

	status := &WatcherStatus{
		Config:    cfg,
		State:     WatcherInitializing,
		StartedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	wm.saveState(key, status)

	go wm.runWatcher(ctx, cfg, status, key)

	log.Printf("[watcher] Started %s %s target_qty=%.6f limit=%.4f expires=%s",
		cfg.Side, cfg.Symbol, cfg.TargetQty, cfg.LimitPrice,
		cfg.ExpiresAt.Format("15:04:05"))

	return nil
}

// StopWatcher cancels a running watcher.
func (wm *WatcherManager) StopWatcher(symbol, side string) {
	wm.mu.Lock()
	defer wm.mu.Unlock()
	key := symbol + ":" + side
	if cancel, ok := wm.watchers[key]; ok {
		cancel()
		delete(wm.watchers, key)
		log.Printf("[watcher] Stopped %s", key)
	}
}

// ListWatchers returns the status of all active watchers.
func (wm *WatcherManager) ListWatchers() []WatcherStatus {
	var results []WatcherStatus
	entries, _ := os.ReadDir(wm.stateDir)
	for _, e := range entries {
		data, err := os.ReadFile(filepath.Join(wm.stateDir, e.Name()))
		if err != nil {
			continue
		}
		var s WatcherStatus
		if json.Unmarshal(data, &s) == nil {
			results = append(results, s)
		}
	}
	return results
}

func (wm *WatcherManager) runWatcher(ctx context.Context, cfg WatcherConfig,
	status *WatcherStatus, key string) {

	defer func() {
		wm.mu.Lock()
		delete(wm.watchers, key)
		wm.mu.Unlock()
	}()

	side := alpaca.Buy
	if cfg.Side == "sell" {
		side = alpaca.Sell
	}

	status.State = WatcherWaiting
	wm.saveState(key, status)

	for {
		select {
		case <-ctx.Done():
			status.State = WatcherDone
			wm.saveState(key, status)
			return
		default:
		}

		// Check expiry
		if time.Now().After(cfg.ExpiresAt) {
			log.Printf("[watcher] %s expired", key)
			// Cancel any pending orders
			wm.trader.cancelOrdersForSymbol(remapSymbol(cfg.Symbol), side)
			status.State = WatcherExpired
			wm.saveState(key, status)
			return
		}

		// Check current position
		currentQty := wm.trader.currentPositionQty(remapSymbol(cfg.Symbol), side)
		status.PositionQty = currentQty
		status.UpdatedAt = time.Now()

		if currentQty >= cfg.TargetQty {
			log.Printf("[watcher] %s position full (%.6f >= %.6f)", key, currentQty, cfg.TargetQty)
			status.State = WatcherPositionOpen
			wm.saveState(key, status)
			return
		}

		// Get current quote
		quote, err := wm.trader.GetLatestQuote(cfg.Symbol)
		if err != nil {
			log.Printf("[watcher] %s quote error: %v", key, err)
			time.Sleep(cfg.PollInterval)
			continue
		}

		// Check if price is within tolerance
		refPrice := quote.BidPrice
		if side == alpaca.Sell {
			refPrice = quote.AskPrice
		}

		if cfg.LimitPrice <= 0 {
			log.Printf("[watcher] %s invalid limit price %.4f", key, cfg.LimitPrice)
			status.State = WatcherError
			status.Error = "invalid limit price"
			wm.saveState(key, status)
			return
		}
		priceDiff := math.Abs(refPrice-cfg.LimitPrice) / cfg.LimitPrice
		if priceDiff > cfg.TolerancePct {
			// Price not within tolerance, keep waiting
			status.State = WatcherWaiting
			wm.saveState(key, status)
			time.Sleep(cfg.PollInterval)
			continue
		}

		// Price within tolerance - enter!
		log.Printf("[watcher] %s triggered! price=%.4f target=%.4f diff=%.4f%%",
			key, refPrice, cfg.LimitPrice, priceDiff*100)

		status.State = WatcherSubmitting
		wm.saveState(key, status)

		remaining := cfg.TargetQty - currentQty

		if cfg.RampOnTrigger {
			// Use ramp to accumulate
			rampMins := cfg.RampMins
			if rampMins <= 0 {
				rampMins = 30
			}
			err = wm.trader.RampIntoPosition(ctx, RampConfig{
				Symbol:       cfg.Symbol,
				Side:         side,
				TargetQty:    cfg.TargetQty,
				DurationMins: rampMins,
			})
			if err != nil {
				log.Printf("[watcher] %s ramp error: %v", key, err)
				status.State = WatcherError
				status.Error = err.Error()
				wm.saveState(key, status)
				time.Sleep(cfg.PollInterval)
				continue
			}
		} else {
			// Single limit order
			_, err = wm.trader.PlaceLimitOrder(cfg.Symbol, remaining, side, cfg.LimitPrice)
			if err != nil {
				log.Printf("[watcher] %s order error: %v", key, err)
				status.State = WatcherError
				status.Error = err.Error()
				wm.saveState(key, status)
				time.Sleep(cfg.PollInterval)
				continue
			}
		}

		status.State = WatcherAwaitingFill
		wm.saveState(key, status)
		time.Sleep(cfg.PollInterval)
	}
}

func (wm *WatcherManager) saveState(key string, status *WatcherStatus) {
	status.UpdatedAt = time.Now()
	data, _ := json.MarshalIndent(status, "", "  ")
	path := filepath.Join(wm.stateDir, key+".json")
	os.WriteFile(path, data, 0644)
}

// ---------- Trade Update Streaming ----------

// StreamTradeUpdates listens for order fill events in the background.
// The handler is called for each trade update (fill, partial_fill, canceled, etc).
func (t *Trader) StreamTradeUpdates(ctx context.Context, handler func(alpaca.TradeUpdate)) {
	t.client.StreamTradeUpdatesInBackground(ctx, handler)
}

// ---------- Hourly Loop (matches Python trade_alpaca_selector pattern) ----------

// HourlyLoopConfig configures the main trading loop.
type HourlyLoopConfig struct {
	Symbols       []string
	PollInterval  time.Duration // How often to check within the hour
	BufferSeconds int           // Seconds after hour to start (default 30)
	OnCycle       func(symbols []string) error // Called each hour with the trading logic
}

// RunHourlyLoop runs a trading function on every hour boundary.
// This is the Go equivalent of trade_alpaca_selector's main loop.
func RunHourlyLoop(ctx context.Context, cfg HourlyLoopConfig) {
	if cfg.BufferSeconds <= 0 {
		cfg.BufferSeconds = 30
	}

	for {
		// Sleep until next hour + buffer
		sleepDur := secondsUntilNextHour(cfg.BufferSeconds)
		log.Printf("[hourly] Sleeping %s until next cycle", sleepDur.Round(time.Second))

		select {
		case <-ctx.Done():
			log.Println("[hourly] Loop cancelled")
			return
		case <-time.After(sleepDur):
		}

		log.Printf("[hourly] Starting cycle for %v", cfg.Symbols)
		if err := cfg.OnCycle(cfg.Symbols); err != nil {
			log.Printf("[hourly] Cycle error: %v", err)
		}
	}
}

func secondsUntilNextHour(bufferSec int) time.Duration {
	now := time.Now()
	nextHour := now.Truncate(time.Hour).Add(time.Hour).Add(time.Duration(bufferSec) * time.Second)
	d := nextHour.Sub(now)
	if d < 0 {
		d += time.Hour
	}
	return d
}

// ---------- Multi-symbol goroutine orchestration ----------

// RunParallelWatchers starts watchers for multiple symbols concurrently.
func RunParallelWatchers(ctx context.Context, trader *Trader, configs []WatcherConfig) *WatcherManager {
	wm := NewWatcherManager(trader)

	// Stream trade updates for logging
	trader.StreamTradeUpdates(ctx, func(tu alpaca.TradeUpdate) {
		log.Printf("[trade-update] %s %s %s qty=%s status=%s",
			tu.Event, tu.Order.Symbol, tu.Order.Side,
			derefDec(tu.Order.Qty).String(), tu.Order.Status)
	})

	for _, cfg := range configs {
		if err := wm.StartWatcher(cfg); err != nil {
			log.Printf("[parallel] Error starting watcher for %s: %v", cfg.Symbol, err)
		}
	}

	return wm
}

// WaitForAllWatchers blocks until all watchers complete or context cancels.
func (wm *WatcherManager) WaitForAllWatchers(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			wm.mu.Lock()
			n := len(wm.watchers)
			wm.mu.Unlock()
			if n == 0 {
				log.Println("[watchers] All watchers completed")
				return
			}
		}
	}
}

// PrintWatcherStatus prints a summary of all watchers.
func (wm *WatcherManager) PrintWatcherStatus() {
	statuses := wm.ListWatchers()
	if len(statuses) == 0 {
		fmt.Println("No active watchers.")
		return
	}
	fmt.Printf("%-12s %-6s %-18s %10s %10s %s\n",
		"Symbol", "Side", "State", "Filled", "Target", "Updated")
	for _, s := range statuses {
		fmt.Printf("%-12s %-6s %-18s %10.4f %10.4f %s\n",
			s.Config.Symbol,
			s.Config.Side,
			s.State,
			s.PositionQty,
			s.Config.TargetQty,
			s.UpdatedAt.Format("15:04:05"),
		)
	}
}
