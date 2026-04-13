package policy

import "math"

// DecodeConfig holds parameters for action decoding, matching BinancePolicyBase.
type DecodeConfig struct {
	PriceOffsetPct    float64
	MinGapPct         float64
	TradeAmountScale  float64
	UseMidpointOffsets bool
	MaxHoldHours      float64
}

func DefaultDecodeConfig() DecodeConfig {
	return DecodeConfig{
		PriceOffsetPct:    0.02,
		MinGapPct:         0.001,
		TradeAmountScale:  100.0,
		UseMidpointOffsets: true,
		MaxHoldHours:      24.0,
	}
}

// DecodedAction is the output of action decoding.
type DecodedAction struct {
	BuyPrice    float64
	SellPrice   float64
	BuyAmount   float64
	SellAmount  float64
	TradeAmount float64
	HoldHours   float64
}

// DecodeActions converts raw model logits into trading actions.
// logits: [buy_price_logit, sell_price_logit, buy_amount_logit, sell_amount_logit, ...]
// refClose: reference close price
// chronosHigh: Chronos2 predicted high
// chronosLow: Chronos2 predicted low
func DecodeActions(logits []float64, refClose, chronosHigh, chronosLow float64, cfg DecodeConfig) DecodedAction {
	if len(logits) < 4 {
		return DecodedAction{}
	}

	ref := math.Max(refClose, 1e-8)
	lowAnchor := math.Min(chronosLow, ref)
	highAnchor := math.Max(chronosHigh, ref)

	buyUnit := sigmoid(logits[0])
	sellUnit := sigmoid(logits[1])

	var buyPrice, sellPrice float64
	if cfg.UseMidpointOffsets {
		buyPrice = lowAnchor + buyUnit*(ref-lowAnchor)
		sellPrice = ref + sellUnit*(highAnchor-ref)
	} else {
		buyOffset := cfg.PriceOffsetPct * buyUnit
		sellOffset := cfg.PriceOffsetPct * sellUnit
		buyPrice = ref * (1 - buyOffset)
		sellPrice = ref * (1 + sellOffset)
	}

	// Gap enforcement
	gap := math.Max(ref*cfg.MinGapPct, 1e-8)
	sellPrice = math.Max(sellPrice, buyPrice+gap)

	buyAmount := sigmoid(logits[2]) * cfg.TradeAmountScale
	sellAmount := sigmoid(logits[3]) * cfg.TradeAmountScale
	tradeAmount := math.Max(buyAmount, sellAmount)

	result := DecodedAction{
		BuyPrice:    buyPrice,
		SellPrice:   sellPrice,
		BuyAmount:   buyAmount,
		SellAmount:  sellAmount,
		TradeAmount: tradeAmount,
	}

	if len(logits) > 4 {
		result.HoldHours = sigmoid(logits[4]) * cfg.MaxHoldHours
	}

	return result
}

// DecodeActionsBatch decodes a batch of logits for a sequence of timesteps.
func DecodeActionsBatch(logits [][]float64, refCloses, chronosHighs, chronosLows []float64, cfg DecodeConfig) []DecodedAction {
	n := len(logits)
	actions := make([]DecodedAction, n)
	for i := 0; i < n; i++ {
		actions[i] = DecodeActions(logits[i], refCloses[i], chronosHighs[i], chronosLows[i], cfg)
	}
	return actions
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1.0 / (1.0 + z)
	}
	z := math.Exp(x)
	return z / (1.0 + z)
}
