package sim

type SimConfig struct {
	MaxLeverage      float64
	CanShort         bool
	MakerFee         float64
	MarginHourlyRate float64
	InitialCash      float64
	FillBufferPct    float64
	MinEdge          float64
	MaxHoldBars      int
	IntensityScale   float64
	DecisionLagBars  int
	OneSidePerBar    bool
}

func DefaultSimConfig() SimConfig {
	return SimConfig{
		MaxLeverage:      2.0,
		MakerFee:         0.001,
		MarginHourlyRate: 0.0,
		InitialCash:      10000.0,
		FillBufferPct:    0.0005,
		MinEdge:          0.0,
		MaxHoldBars:      0,
		IntensityScale:   1.0,
	}
}

type SimResult struct {
	TotalReturn     float64
	Sortino         float64
	MaxDrawdown     float64
	FinalEquity     float64
	NumTrades       int
	MarginCostTotal float64
	EquityCurve     []float64
}

type Bar struct {
	Open, High, Low, Close float64
}

type Action struct {
	BuyPrice, SellPrice   float64
	BuyAmount, SellAmount float64
}

type SharedCashConfig struct {
	MakerFee         float64
	InitialCash      float64
	MaxHoldHours     int
	FillBufferBps    float64
	DecisionLagBars  int
	OneSidePerBar    bool
	ForceCloseOnHold bool
	TickSize         float64
	StepSize         float64
	MinNotional      float64
	MinQty           float64
}

func DefaultSharedCashConfig() SharedCashConfig {
	return SharedCashConfig{
		MakerFee:         0.001,
		InitialCash:      10000.0,
		MaxHoldHours:     24,
		FillBufferBps:    5.0,
		DecisionLagBars:  0,
		OneSidePerBar:    true,
		ForceCloseOnHold: true,
	}
}

type TradeRecord struct {
	Timestamp      int64
	Symbol         string
	Side           string
	Price          float64
	Quantity       float64
	Notional       float64
	Fee            float64
	CashAfter      float64
	InventoryAfter float64
	RealizedPnl    float64
	Reason         string
}

type SharedCashResult struct {
	EquityCurve []float64
	Trades      []TradeRecord
	Metrics     map[string]float64
}
