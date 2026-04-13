package sweep

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"binancego/sim"
)

// CMAESConfig configures the CMA-ES optimizer.
type CMAESConfig struct {
	PopSize      int
	MaxIter      int
	Sigma0       float64 // initial step size
	TargetMetric string  // "sortino" or "total_return"
	NumWorkers   int
	Verbose      bool
}

func DefaultCMAESConfig() CMAESConfig {
	return CMAESConfig{
		PopSize:      20,
		MaxIter:      50,
		Sigma0:       0.3,
		TargetMetric: "sortino",
		Verbose:      true,
	}
}

// ParamBounds defines the search space for each parameter.
type ParamBounds struct {
	Name string
	Min  float64
	Max  float64
}

var DefaultParamBounds = []ParamBounds{
	{"intensity_scale", 0.5, 3.0},
	{"fill_buffer_pct", 0.0, 0.002},
	{"min_edge", 0.0, 0.02},
	{"max_hold_bars", 6, 48},
	{"maker_fee", 0.0005, 0.002},
}

// CMAESResult is the output of optimization.
type CMAESResult struct {
	BestParams  map[string]float64
	BestMetric  float64
	Iterations  int
	Evaluations int
}

// OptimizeCMAES runs CMA-ES to find optimal sim config parameters.
func OptimizeCMAES(
	bars []sim.Bar,
	actions []sim.Action,
	baseCfg sim.SimConfig,
	bounds []ParamBounds,
	cmaConfig CMAESConfig,
) CMAESResult {
	dim := len(bounds)
	lambda := cmaConfig.PopSize
	if lambda <= 0 {
		lambda = 4 + int(3*math.Log(float64(dim)))
	}
	mu := lambda / 2

	// Initialize
	mean := make([]float64, dim)
	for i := range mean {
		mean[i] = 0.5 // normalized [0,1] space
	}
	sigma := cmaConfig.Sigma0

	// Weights for recombination
	weights := make([]float64, mu)
	sumW := 0.0
	for i := range weights {
		weights[i] = math.Log(float64(mu)+0.5) - math.Log(float64(i+1))
		sumW += weights[i]
	}
	for i := range weights {
		weights[i] /= sumW
	}

	// Covariance matrix (diagonal approximation for simplicity)
	C := make([]float64, dim)
	for i := range C {
		C[i] = 1.0
	}

	pool := NewWorkerPool(cmaConfig.NumWorkers)

	bestMetric := math.Inf(-1)
	bestParams := make(map[string]float64)
	totalEvals := 0

	for iter := 0; iter < cmaConfig.MaxIter; iter++ {
		// Sample population
		population := make([][]float64, lambda)
		for i := range population {
			population[i] = make([]float64, dim)
			for j := range population[i] {
				population[i][j] = mean[j] + sigma*math.Sqrt(C[j])*rand.NormFloat64()
				population[i][j] = clamp01(population[i][j])
			}
		}

		// Evaluate
		var jobs []SimJob
		for i, p := range population {
			cfg := applyParams(baseCfg, p, bounds)
			jobs = append(jobs, SimJob{
				ID:      i,
				Bars:    bars,
				Actions: actions,
				Config:  cfg,
			})
		}
		results := pool.Run(jobs)
		totalEvals += lambda

		// Extract fitness
		type scored struct {
			idx    int
			params []float64
			value  float64
		}
		var scored_pop []scored
		for i, r := range results {
			var value float64
			switch cmaConfig.TargetMetric {
			case "sortino":
				value = r.Result.Sortino
			case "total_return":
				value = r.Result.TotalReturn
			default:
				value = r.Result.Sortino
			}
			scored_pop = append(scored_pop, scored{i, population[i], value})
		}

		sort.Slice(scored_pop, func(i, j int) bool {
			return scored_pop[i].value > scored_pop[j].value
		})

		// Update best
		if scored_pop[0].value > bestMetric {
			bestMetric = scored_pop[0].value
			bestParams = denormParams(scored_pop[0].params, bounds)
		}

		if cmaConfig.Verbose {
			fmt.Printf("  CMA-ES iter=%d best=%.4f pop_best=%.4f sigma=%.4f\n",
				iter, bestMetric, scored_pop[0].value, sigma)
		}

		// Update mean (weighted recombination of top-mu)
		newMean := make([]float64, dim)
		for i := 0; i < mu; i++ {
			for j := 0; j < dim; j++ {
				newMean[j] += weights[i] * scored_pop[i].params[j]
			}
		}

		// Update covariance (simplified rank-mu update)
		for j := 0; j < dim; j++ {
			var sumSq float64
			for i := 0; i < mu; i++ {
				diff := scored_pop[i].params[j] - mean[j]
				sumSq += weights[i] * diff * diff
			}
			C[j] = 0.8*C[j] + 0.2*sumSq/(sigma*sigma+1e-10)
		}

		// Sigma adaptation (CSA simplified)
		meanShift := 0.0
		for j := 0; j < dim; j++ {
			d := newMean[j] - mean[j]
			meanShift += d * d
		}
		meanShift = math.Sqrt(meanShift)
		expectedStep := math.Sqrt(float64(dim)) * sigma
		if expectedStep > 0 {
			ratio := meanShift / expectedStep
			sigma *= math.Exp((ratio - 1) * 0.3)
		}
		sigma = math.Max(sigma, 0.01)
		sigma = math.Min(sigma, 2.0)

		mean = newMean
	}

	return CMAESResult{
		BestParams:  bestParams,
		BestMetric:  bestMetric,
		Iterations:  cmaConfig.MaxIter,
		Evaluations: totalEvals,
	}
}

func applyParams(base sim.SimConfig, normalized []float64, bounds []ParamBounds) sim.SimConfig {
	cfg := base
	for i, b := range bounds {
		val := b.Min + normalized[i]*(b.Max-b.Min)
		switch b.Name {
		case "intensity_scale":
			cfg.IntensityScale = val
		case "fill_buffer_pct":
			cfg.FillBufferPct = val
		case "min_edge":
			cfg.MinEdge = val
		case "max_hold_bars":
			cfg.MaxHoldBars = int(math.Round(val))
		case "maker_fee":
			cfg.MakerFee = val
		}
	}
	return cfg
}

func denormParams(normalized []float64, bounds []ParamBounds) map[string]float64 {
	params := make(map[string]float64)
	for i, b := range bounds {
		params[b.Name] = b.Min + normalized[i]*(b.Max-b.Min)
	}
	return params
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
