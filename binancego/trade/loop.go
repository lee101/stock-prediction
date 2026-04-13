package trade

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"binancego/data"
	"binancego/policy"
)

type TradingLoop struct {
	Client       *BinanceClient
	Guard        *DeathSpiralGuard
	Singleton    *SingletonLock
	Model        *policy.ONNXModel
	Forecasts    map[string]*data.ForecastCache
	Normalizer   *data.FeatureNormalizer
	Symbols      []string
	Interval     time.Duration
	DecodeConfig policy.DecodeConfig
	SeqLen       int
	DataRoot     string
}

type TradingLoopConfig struct {
	StateDir     string
	ModelPath    string
	ForecastDir  string
	MetaPath     string
	Symbols      []string
	SeqLen       int
	DataRoot     string
	Interval     time.Duration
}

func NewTradingLoop(cfg TradingLoopConfig) (*TradingLoop, error) {
	lock, err := AcquireSingleton(cfg.StateDir)
	if err != nil {
		return nil, err
	}

	model, err := policy.LoadONNX(cfg.ModelPath)
	if err != nil {
		lock.Release()
		return nil, fmt.Errorf("load model: %w", err)
	}

	var forecasts map[string]*data.ForecastCache
	if cfg.ForecastDir != "" {
		forecasts, err = data.LoadForecastDir(cfg.ForecastDir)
		if err != nil {
			log.Printf("warn: no forecast cache: %v", err)
		}
	}

	var norm *data.FeatureNormalizer
	if cfg.MetaPath != "" {
		norm, err = data.LoadNormalizerFromMeta(cfg.MetaPath)
		if err != nil {
			log.Printf("warn: no normalizer: %v", err)
		}
	}

	interval := cfg.Interval
	if interval == 0 {
		interval = 1 * time.Hour
	}

	return &TradingLoop{
		Client:       NewBinanceClient(),
		Guard:        NewDeathSpiralGuard(cfg.StateDir),
		Singleton:    lock,
		Model:        model,
		Forecasts:    forecasts,
		Normalizer:   norm,
		Symbols:      cfg.Symbols,
		Interval:     interval,
		DecodeConfig: policy.DefaultDecodeConfig(),
		SeqLen:       cfg.SeqLen,
		DataRoot:     cfg.DataRoot,
	}, nil
}

func (tl *TradingLoop) Run() {
	defer tl.Singleton.Release()
	defer tl.Model.Close()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	// Claim writer lease
	lease, err := tl.Client.ClaimWriter(1800)
	if err != nil {
		log.Fatalf("claim writer: %v", err)
	}
	log.Printf("writer claimed, expires=%s", lease.ExpiresAt)

	// Heartbeat goroutine
	heartbeatDone := make(chan struct{})
	go func() {
		defer close(heartbeatDone)
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if err := tl.Client.HeartbeatWriter(60); err != nil {
					log.Printf("heartbeat failed: %v", err)
				}
			case <-heartbeatDone:
				return
			}
		}
	}()

	log.Printf("trading loop: %d symbols, interval=%s", len(tl.Symbols), tl.Interval)
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
	log.Printf("tick: %d symbols", len(tl.Symbols))

	prices, err := tl.Client.RefreshPrices(tl.Symbols)
	if err != nil {
		log.Printf("refresh prices error: %v", err)
		return
	}

	acct, err := tl.Client.GetAccount()
	if err != nil {
		log.Printf("get account error: %v", err)
		return
	}
	log.Printf("cash=%.2f positions=%d", acct.Cash, len(acct.Positions))

	for _, sym := range tl.Symbols {
		quote, ok := prices[sym]
		if !ok {
			log.Printf("  %s: no quote", sym)
			continue
		}

		features, err := tl.buildFeatures(sym)
		if err != nil {
			log.Printf("  %s: features error: %v", sym, err)
			continue
		}

		logits, err := tl.Model.InferSequence(features)
		if err != nil {
			log.Printf("  %s: inference error: %v", sym, err)
			continue
		}

		lastLogits := logits[len(logits)-1]
		refClose := quote.LastPrice
		high, low := refClose*1.01, refClose*0.99

		if fc, ok := tl.Forecasts[sym]; ok {
			now := time.Now().UTC()
			if row, found := fc.GetForecast(now.Unix()); found {
				high = row.PredictedHighP50
				low = row.PredictedLowP50
			}
		}

		action := policy.DecodeActions(lastLogits, refClose, high, low, tl.DecodeConfig)

		pos := acct.Positions[sym]
		tl.executeAction(sym, action, pos, quote)
	}
}

func (tl *TradingLoop) buildFeatures(symbol string) ([][]float32, error) {
	csvPath := fmt.Sprintf("%s/%s.csv", tl.DataRoot, symbol)
	bars, err := data.LoadCSV(csvPath)
	if err != nil {
		return nil, err
	}

	if len(bars) < tl.SeqLen {
		return nil, fmt.Errorf("not enough bars: %d < %d", len(bars), tl.SeqLen)
	}
	bars = bars[len(bars)-tl.SeqLen:]

	feats := data.ComputeFeatures(bars)
	featureNames := []string{"return_1h", "volatility_24h", "volume_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "high_low_range"}

	n := len(bars)
	result := make([][]float32, n)
	for i := 0; i < n; i++ {
		row := make([]float32, len(featureNames))
		for j, name := range featureNames {
			val := feats[name][i]
			if tl.Normalizer != nil {
				val = tl.Normalizer.Normalize(name, val)
			}
			row[j] = float32(val)
		}
		result[i] = row
	}
	return result, nil
}

func (tl *TradingLoop) executeAction(symbol string, action policy.DecodedAction, pos PositionState, quote QuotePayload) {
	if pos.Qty > 0 && action.SellPrice > 0 && action.SellAmount > 50 {
		sellQty := pos.Qty * (action.SellAmount / 100.0)
		if sellQty > 0.0001 {
			if err := tl.Guard.CheckSell(symbol, action.SellPrice); err != nil {
				log.Printf("  %s: %v", symbol, err)
				return
			}
			resp, err := tl.Client.SubmitOrder(symbol, "sell", sellQty, action.SellPrice, map[string]interface{}{
				"strategy": "go-onnx",
			})
			if err != nil {
				log.Printf("  %s: sell error: %v", symbol, err)
			} else {
				log.Printf("  %s: SELL %.6f @ %.2f (order=%s)", symbol, sellQty, action.SellPrice, resp.Order.ID)
			}
		}
	}

	if action.BuyPrice > 0 && action.BuyAmount > 50 {
		notional := action.BuyAmount
		buyQty := notional / quote.AskPrice
		if buyQty > 0.0001 {
			resp, err := tl.Client.SubmitOrder(symbol, "buy", buyQty, action.BuyPrice, map[string]interface{}{
				"strategy": "go-onnx",
			})
			if err != nil {
				log.Printf("  %s: buy error: %v", symbol, err)
			} else {
				log.Printf("  %s: BUY %.6f @ %.2f (order=%s)", symbol, buyQty, action.BuyPrice, resp.Order.ID)
				tl.Guard.RecordBuy(symbol, action.BuyPrice)
			}
		}
	}
}
