package sweep

import (
	"runtime"
	"sync"

	"binancego/sim"
)

// SimJob represents a single simulation job.
type SimJob struct {
	ID      int
	Bars    []sim.Bar
	Actions []sim.Action
	Config  sim.SimConfig
	Label   string // e.g. "epoch_005/BTCUSD/lag=2/30d"
}

// SimJobResult is the result of a simulation job.
type SimJobResult struct {
	ID     int
	Label  string
	Result sim.SimResult
}

// WorkerPool runs simulation jobs in parallel using goroutines.
type WorkerPool struct {
	NumWorkers int
}

// NewWorkerPool creates a pool sized to NumCPU by default.
func NewWorkerPool(numWorkers int) *WorkerPool {
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}
	return &WorkerPool{NumWorkers: numWorkers}
}

// Run executes all jobs and returns results (order preserved by ID).
func (p *WorkerPool) Run(jobs []SimJob) []SimJobResult {
	results := make([]SimJobResult, len(jobs))
	jobCh := make(chan int, len(jobs))

	var wg sync.WaitGroup
	for w := 0; w < p.NumWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobCh {
				j := jobs[idx]
				r := sim.Simulate(j.Bars, j.Actions, j.Config)
				results[idx] = SimJobResult{
					ID:     j.ID,
					Label:  j.Label,
					Result: r,
				}
			}
		}()
	}

	for i := range jobs {
		jobCh <- i
	}
	close(jobCh)
	wg.Wait()

	return results
}

// SharedCashJob is a multi-symbol simulation job.
type SharedCashJob struct {
	ID      int
	Bars    []sim.SymbolBar
	Actions []sim.SymbolAction
	Config  sim.SharedCashConfig
	Label   string
}

// SharedCashJobResult is the result of a shared-cash sim job.
type SharedCashJobResult struct {
	ID     int
	Label  string
	Result sim.SharedCashResult
}

// RunSharedCash executes shared-cash jobs in parallel.
func (p *WorkerPool) RunSharedCash(jobs []SharedCashJob) []SharedCashJobResult {
	results := make([]SharedCashJobResult, len(jobs))
	jobCh := make(chan int, len(jobs))

	var wg sync.WaitGroup
	for w := 0; w < p.NumWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobCh {
				j := jobs[idx]
				r := sim.SimulateSharedCash(j.Bars, j.Actions, j.Config)
				results[idx] = SharedCashJobResult{
					ID:     j.ID,
					Label:  j.Label,
					Result: r,
				}
			}
		}()
	}

	for i := range jobs {
		jobCh <- i
	}
	close(jobCh)
	wg.Wait()

	return results
}
