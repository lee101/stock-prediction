package trade

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSingletonLock(t *testing.T) {
	tmp := t.TempDir()
	lock, err := AcquireSingleton(tmp)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer lock.Release()

	// Second acquire should fail
	_, err = AcquireSingleton(tmp)
	if err == nil {
		t.Error("expected second acquire to fail")
	}
}

func TestSingletonLockRelease(t *testing.T) {
	tmp := t.TempDir()
	lock1, err := AcquireSingleton(tmp)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	lock1.Release()

	// Should be able to acquire again after release
	lock2, err := AcquireSingleton(tmp)
	if err != nil {
		t.Fatalf("re-acquire after release: %v", err)
	}
	lock2.Release()
}

func TestSingletonOverride(t *testing.T) {
	os.Setenv("ALPACA_SINGLETON_OVERRIDE", "1")
	defer os.Unsetenv("ALPACA_SINGLETON_OVERRIDE")

	tmp := t.TempDir()
	lock1, err := AcquireSingleton(tmp)
	if err != nil {
		t.Fatalf("acquire with override: %v", err)
	}
	defer lock1.Release()

	lock2, err := AcquireSingleton(tmp)
	if err != nil {
		t.Fatalf("second acquire with override should succeed: %v", err)
	}
	defer lock2.Release()
}

func TestDeathSpiralGuard(t *testing.T) {
	tmp := t.TempDir()
	g := NewDeathSpiralGuard(tmp)

	g.RecordBuy("BTCUSD", 50000)

	// Sell at 50000 should be fine
	if err := g.CheckSell("BTCUSD", 50000); err != nil {
		t.Errorf("sell at buy price should be OK: %v", err)
	}

	// Sell 30bps below should be fine
	if err := g.CheckSell("BTCUSD", 49850); err != nil {
		t.Errorf("sell 30bps below should be OK: %v", err)
	}

	// Sell 60bps below should be rejected
	if err := g.CheckSell("BTCUSD", 49700); err == nil {
		t.Error("sell 60bps below should be rejected")
	}

	// Unknown symbol should be fine
	if err := g.CheckSell("ETHUSD", 3000); err != nil {
		t.Errorf("unknown symbol should be OK: %v", err)
	}
}

func TestDeathSpiralPersistence(t *testing.T) {
	tmp := t.TempDir()
	g1 := NewDeathSpiralGuard(tmp)
	g1.RecordBuy("BTCUSD", 50000)

	// Load a new guard from the same directory
	g2 := NewDeathSpiralGuard(tmp)
	if err := g2.CheckSell("BTCUSD", 49700); err == nil {
		t.Error("persisted guard should still reject death spiral sell")
	}

	// Verify file exists
	buysPath := filepath.Join(tmp, "binance_singleton", "binance_live_writer_buys.json")
	if _, err := os.Stat(buysPath); err != nil {
		t.Errorf("buys file not found: %v", err)
	}
}

func TestDeathSpiralOverride(t *testing.T) {
	os.Setenv("ALPACA_DEATH_SPIRAL_OVERRIDE", "1")
	defer os.Unsetenv("ALPACA_DEATH_SPIRAL_OVERRIDE")

	tmp := t.TempDir()
	g := NewDeathSpiralGuard(tmp)
	g.RecordBuy("BTCUSD", 50000)

	if err := g.CheckSell("BTCUSD", 40000); err != nil {
		t.Errorf("override should bypass guard: %v", err)
	}
}
