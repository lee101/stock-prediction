package trade

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"time"
)

// SingletonLock ensures only one live trading instance runs at a time.
// Matches src/alpaca_singleton.py:enforce_live_singleton.
type SingletonLock struct {
	lockFile *os.File
	path     string
}

// AcquireSingleton takes an fcntl lock. Returns error if another instance holds it.
func AcquireSingleton(stateDir string) (*SingletonLock, error) {
	if os.Getenv("ALPACA_SINGLETON_OVERRIDE") == "1" {
		fmt.Println("WARNING: ALPACA_SINGLETON_OVERRIDE=1, skipping singleton lock")
		return &SingletonLock{}, nil
	}

	lockDir := filepath.Join(stateDir, "account_locks")
	if err := os.MkdirAll(lockDir, 0755); err != nil {
		return nil, fmt.Errorf("create lock dir: %w", err)
	}

	lockPath := filepath.Join(lockDir, "binance_live_writer.lock")
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, fmt.Errorf("open lock file: %w", err)
	}

	err = syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("another live trading instance is running (lock held on %s)", lockPath)
	}

	// Write PID info
	info := fmt.Sprintf("pid=%d host=%s time=%s\n", os.Getpid(), hostname(), time.Now().UTC().Format(time.RFC3339))
	f.Truncate(0)
	f.Seek(0, 0)
	f.WriteString(info)

	return &SingletonLock{lockFile: f, path: lockPath}, nil
}

// Release releases the singleton lock.
func (s *SingletonLock) Release() {
	if s.lockFile != nil {
		syscall.Flock(int(s.lockFile.Fd()), syscall.LOCK_UN)
		s.lockFile.Close()
	}
}

func hostname() string {
	h, _ := os.Hostname()
	return h
}

// DeathSpiralGuard prevents selling at a loss below the most recent buy price.
// Matches alpaca_wrapper.guard_sell_against_death_spiral.
type DeathSpiralGuard struct {
	mu       sync.Mutex
	buys     map[string]buyRecord
	filePath string
	ttl      time.Duration
}

type buyRecord struct {
	Price     float64   `json:"price"`
	Timestamp time.Time `json:"timestamp"`
}

// NewDeathSpiralGuard creates a guard with 3-day TTL.
func NewDeathSpiralGuard(stateDir string) *DeathSpiralGuard {
	g := &DeathSpiralGuard{
		buys:     make(map[string]buyRecord),
		filePath: filepath.Join(stateDir, "binance_singleton", "binance_live_writer_buys.json"),
		ttl:      72 * time.Hour,
	}
	g.load()
	return g
}

// RecordBuy records a buy price for a symbol.
func (g *DeathSpiralGuard) RecordBuy(symbol string, price float64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.buys[symbol] = buyRecord{Price: price, Timestamp: time.Now().UTC()}
	g.save()
}

// CheckSell validates that a sell price is not too far below the last buy.
// Returns error if sell would trigger death spiral protection (>50bps below buy).
func (g *DeathSpiralGuard) CheckSell(symbol string, sellPrice float64) error {
	if os.Getenv("ALPACA_DEATH_SPIRAL_OVERRIDE") == "1" {
		fmt.Println("WARNING: ALPACA_DEATH_SPIRAL_OVERRIDE=1, bypassing death spiral guard")
		return nil
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	rec, ok := g.buys[symbol]
	if !ok {
		return nil // no recent buy
	}
	if time.Since(rec.Timestamp) > g.ttl {
		delete(g.buys, symbol)
		return nil // expired
	}

	threshold := rec.Price * (1 - 0.005) // 50bps
	if sellPrice < threshold {
		return fmt.Errorf(
			"death spiral guard: sell %.4f is %.1fbps below last buy %.4f for %s",
			sellPrice, (1-sellPrice/rec.Price)*10000, rec.Price, symbol,
		)
	}
	return nil
}

func (g *DeathSpiralGuard) load() {
	data, err := os.ReadFile(g.filePath)
	if err != nil {
		return
	}
	var buys map[string]buyRecord
	if err := json.Unmarshal(data, &buys); err != nil {
		return
	}
	now := time.Now().UTC()
	for k, v := range buys {
		if now.Sub(v.Timestamp) <= g.ttl {
			g.buys[k] = v
		}
	}
}

func (g *DeathSpiralGuard) save() {
	dir := filepath.Dir(g.filePath)
	os.MkdirAll(dir, 0755)
	data, err := json.MarshalIndent(g.buys, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(g.filePath, data, 0644)
}
