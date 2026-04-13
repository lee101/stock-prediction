package trade

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// BinanceClient communicates with the Python binance_trading_server.
type BinanceClient struct {
	BaseURL    string
	AuthToken  string
	AccountID  string
	HTTPClient *http.Client
}

// NewBinanceClient creates a client from env vars.
func NewBinanceClient() *BinanceClient {
	baseURL := os.Getenv("BINANCE_TRADING_SERVER_URL")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8060"
	}
	return &BinanceClient{
		BaseURL:   baseURL,
		AuthToken: os.Getenv("BINANCE_TRADING_SERVER_AUTH_TOKEN"),
		AccountID: os.Getenv("BINANCE_ACCOUNT_ID"),
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// OrderRequest is the payload for placing an order.
type OrderRequest struct {
	Symbol   string  `json:"symbol"`
	Side     string  `json:"side"` // "BUY" or "SELL"
	Type     string  `json:"type"` // "LIMIT" or "MARKET"
	Quantity float64 `json:"quantity"`
	Price    float64 `json:"price,omitempty"`
}

// OrderResponse is returned after placing an order.
type OrderResponse struct {
	OrderID string  `json:"order_id"`
	Status  string  `json:"status"`
	Symbol  string  `json:"symbol"`
	Side    string  `json:"side"`
	Price   float64 `json:"price"`
	Qty     float64 `json:"qty"`
}

// Position represents an open position.
type Position struct {
	Symbol   string  `json:"symbol"`
	Quantity float64 `json:"quantity"`
	EntryPx  float64 `json:"entry_price"`
	UnrealPnl float64 `json:"unrealized_pnl"`
}

// Quote is a market quote.
type Quote struct {
	Symbol string  `json:"symbol"`
	Bid    float64 `json:"bid"`
	Ask    float64 `json:"ask"`
	Last   float64 `json:"last"`
}

// PlaceOrder submits an order to the trading server.
func (c *BinanceClient) PlaceOrder(req OrderRequest) (*OrderResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/accounts/%s/orders", c.BaseURL, c.AccountID)
	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.AuthToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.AuthToken)
	}

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("place order: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("order failed (%d): %s", resp.StatusCode, string(body))
	}

	var result OrderResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &result, nil
}

// GetPositions returns all open positions.
func (c *BinanceClient) GetPositions() ([]Position, error) {
	url := fmt.Sprintf("%s/accounts/%s/positions", c.BaseURL, c.AccountID)
	httpReq, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	if c.AuthToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.AuthToken)
	}

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("get positions: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("positions failed (%d): %s", resp.StatusCode, string(body))
	}

	var positions []Position
	if err := json.NewDecoder(resp.Body).Decode(&positions); err != nil {
		return nil, fmt.Errorf("decode positions: %w", err)
	}
	return positions, nil
}

// GetQuote returns the latest quote for a symbol.
func (c *BinanceClient) GetQuote(symbol string) (*Quote, error) {
	url := fmt.Sprintf("%s/accounts/%s/quote/%s", c.BaseURL, c.AccountID, symbol)
	httpReq, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	if c.AuthToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.AuthToken)
	}

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("get quote: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("quote failed (%d): %s", resp.StatusCode, string(body))
	}

	var quote Quote
	if err := json.NewDecoder(resp.Body).Decode(&quote); err != nil {
		return nil, fmt.Errorf("decode quote: %w", err)
	}
	return &quote, nil
}
