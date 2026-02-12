package main

import (
	"fmt"
	"strings"

	"github.com/alpacahq/alpaca-trade-api-go/v3/alpaca"
	"github.com/shopspring/decimal"
)

// Trader wraps the Alpaca trading client with convenience methods
// mirroring alpaca_wrapper.py functionality.
type Trader struct {
	client *alpaca.Client
	cfg    *Config
}

func NewTrader(cfg *Config) *Trader {
	client := alpaca.NewClient(alpaca.ClientOpts{
		APIKey:    cfg.APIKeyID,
		APISecret: cfg.SecretKey,
		BaseURL:   cfg.BaseURL,
	})
	return &Trader{client: client, cfg: cfg}
}

// ---------- Account ----------

func (t *Trader) GetAccount() (*alpaca.Account, error) {
	return t.client.GetAccount()
}

func (t *Trader) GetClock() (*alpaca.Clock, error) {
	return t.client.GetClock()
}

// ---------- Positions ----------

func (t *Trader) GetPositions() ([]alpaca.Position, error) {
	return t.client.GetPositions()
}

func (t *Trader) GetPosition(symbol string) (*alpaca.Position, error) {
	return t.client.GetPosition(remapSymbol(symbol))
}

func (t *Trader) ClosePosition(symbol string) (*alpaca.Order, error) {
	return t.client.ClosePosition(remapSymbol(symbol), alpaca.ClosePositionRequest{})
}

func (t *Trader) CloseAllPositions() ([]alpaca.Order, error) {
	return t.client.CloseAllPositions(alpaca.CloseAllPositionsRequest{CancelOrders: true})
}

// ---------- Orders ----------

func (t *Trader) PlaceMarketOrder(symbol string, qty float64, side alpaca.Side) (*alpaca.Order, error) {
	sym := remapSymbol(symbol)
	qtyDec := decimal.NewFromFloat(qty)
	tif := timeInForce(sym, qtyDec)

	return t.client.PlaceOrder(alpaca.PlaceOrderRequest{
		Symbol:      sym,
		Qty:         &qtyDec,
		Side:        side,
		Type:        alpaca.Market,
		TimeInForce: tif,
	})
}

func (t *Trader) PlaceLimitOrder(symbol string, qty float64, side alpaca.Side, price float64) (*alpaca.Order, error) {
	sym := remapSymbol(symbol)
	qtyDec := decimal.NewFromFloat(qty)
	priceDec := decimal.NewFromFloat(price)
	tif := timeInForce(sym, qtyDec)

	return t.client.PlaceOrder(alpaca.PlaceOrderRequest{
		Symbol:      sym,
		Qty:         &qtyDec,
		Side:        side,
		Type:        alpaca.Limit,
		TimeInForce: tif,
		LimitPrice:  &priceDec,
	})
}

func (t *Trader) PlaceTrailingStopOrder(symbol string, qty float64, side alpaca.Side, trailPercent float64) (*alpaca.Order, error) {
	sym := remapSymbol(symbol)
	qtyDec := decimal.NewFromFloat(qty)
	trailDec := decimal.NewFromFloat(trailPercent)
	tif := timeInForce(sym, qtyDec)

	return t.client.PlaceOrder(alpaca.PlaceOrderRequest{
		Symbol:       sym,
		Qty:          &qtyDec,
		Side:         side,
		Type:         alpaca.TrailingStop,
		TrailPercent: &trailDec,
		TimeInForce:  tif,
	})
}

func (t *Trader) GetOrders(status string, limit int) ([]alpaca.Order, error) {
	req := alpaca.GetOrdersRequest{
		Status: status,
		Limit:  limit,
	}
	return t.client.GetOrders(req)
}

func (t *Trader) CancelOrder(orderID string) error {
	return t.client.CancelOrder(orderID)
}

func (t *Trader) CancelAllOrders() error {
	return t.client.CancelAllOrders()
}

// GetAsset checks if a symbol is tradable.
func (t *Trader) GetAsset(symbol string) (*alpaca.Asset, error) {
	return t.client.GetAsset(remapSymbol(symbol))
}

// ---------- Helpers ----------

// isCrypto returns true for crypto symbols like BTCUSD, BTC/USD, etc.
func isCrypto(symbol string) bool {
	s := strings.ToUpper(strings.ReplaceAll(symbol, "/", ""))
	cryptoSuffixes := []string{"USD", "USDT", "USDC"}
	cryptoPrefixes := []string{
		"BTC", "ETH", "SOL", "LINK", "UNI", "AAVE", "AVAX", "DOT",
		"DOGE", "LTC", "ALGO", "XRP", "MATIC", "ATOM", "FIL", "NEAR",
		"ADA", "SHIB", "CRV", "MKR", "SUSHI", "BAT", "GRT", "YFI",
	}
	for _, suffix := range cryptoSuffixes {
		if strings.HasSuffix(s, suffix) {
			base := strings.TrimSuffix(s, suffix)
			for _, prefix := range cryptoPrefixes {
				if base == prefix {
					return true
				}
			}
		}
	}
	return false
}

// remapSymbol converts e.g. "BTCUSD" -> "BTC/USD" for Alpaca API.
func remapSymbol(symbol string) string {
	s := strings.ToUpper(strings.TrimSpace(symbol))
	// Already has slash
	if strings.Contains(s, "/") {
		return s
	}
	if !isCrypto(s) {
		return s
	}
	// Try common quote currencies
	for _, quote := range []string{"USDT", "USDC", "USD"} {
		if strings.HasSuffix(s, quote) {
			base := strings.TrimSuffix(s, quote)
			return base + "/" + quote
		}
	}
	return s
}

// unremapSymbol converts e.g. "BTC/USD" -> "BTCUSD".
func unremapSymbol(symbol string) string {
	return strings.ReplaceAll(symbol, "/", "")
}

// timeInForce returns the appropriate TIF:
// crypto -> GTC, fractional stock -> Day, whole stock -> GTC
func timeInForce(symbol string, qty decimal.Decimal) alpaca.TimeInForce {
	if isCrypto(symbol) {
		return alpaca.GTC
	}
	// Fractional shares require Day TIF on Alpaca
	if !qty.Equal(qty.Truncate(0)) {
		return alpaca.Day
	}
	return alpaca.GTC
}

// derefDec safely dereferences a *decimal.Decimal, returning Zero if nil.
func derefDec(d *decimal.Decimal) decimal.Decimal {
	if d == nil {
		return decimal.Zero
	}
	return *d
}

// formatMoney formats a decimal as $X,XXX.XX
func formatMoney(d decimal.Decimal) string {
	f, _ := d.Float64()
	if f < 0 {
		return fmt.Sprintf("-$%s", commify(-f))
	}
	return fmt.Sprintf("$%s", commify(f))
}

func commify(f float64) string {
	s := fmt.Sprintf("%.2f", f)
	parts := strings.Split(s, ".")
	intPart := parts[0]
	decPart := parts[1]

	// Insert commas
	n := len(intPart)
	if n <= 3 {
		return intPart + "." + decPart
	}
	var result []byte
	for i, c := range intPart {
		if i > 0 && (n-i)%3 == 0 {
			result = append(result, ',')
		}
		result = append(result, byte(c))
	}
	return string(result) + "." + decPart
}
