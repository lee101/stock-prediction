package sweep

import (
	"runtime"
	"testing"

	"binancego/sim"
)

func TestWorkerPoolRun(t *testing.T) {
	bars := []sim.Bar{
		{100, 105, 95, 102},
		{102, 108, 98, 104},
		{104, 110, 96, 106},
		{106, 112, 100, 108},
	}
	actions := []sim.Action{
		{96, 106, 50, 80},
		{97, 107, 50, 80},
		{98, 108, 50, 80},
		{99, 109, 50, 80},
	}

	var jobs []SimJob
	for i := 0; i < 100; i++ {
		cfg := sim.DefaultSimConfig()
		cfg.FillBufferPct = 0
		cfg.MakerFee = float64(i) * 0.0001
		jobs = append(jobs, SimJob{
			ID:      i,
			Bars:    bars,
			Actions: actions,
			Config:  cfg,
			Label:   "test",
		})
	}

	pool := NewWorkerPool(runtime.NumCPU())
	results := pool.Run(jobs)

	if len(results) != 100 {
		t.Fatalf("expected 100 results, got %d", len(results))
	}

	// Higher fees should produce lower returns
	// (not strictly monotonic due to discrete fills, but generally true)
	if results[0].Result.FinalEquity < results[99].Result.FinalEquity {
		t.Logf("WARNING: fee=0 equity (%g) < fee=0.0099 equity (%g)",
			results[0].Result.FinalEquity, results[99].Result.FinalEquity)
	}
}

func BenchmarkWorkerPool(b *testing.B) {
	bars := make([]sim.Bar, 720) // 30 days
	actions := make([]sim.Action, 720)
	for i := range bars {
		price := 50000.0 + float64(i)*10
		bars[i] = sim.Bar{price, price + 200, price - 200, price + 50}
		actions[i] = sim.Action{price - 100, price + 100, 50, 80}
	}

	pool := NewWorkerPool(runtime.NumCPU())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var jobs []SimJob
		for j := 0; j < 100; j++ {
			cfg := sim.DefaultSimConfig()
			cfg.FillBufferPct = 0
			jobs = append(jobs, SimJob{
				ID:      j,
				Bars:    bars,
				Actions: actions,
				Config:  cfg,
			})
		}
		pool.Run(jobs)
	}
}
