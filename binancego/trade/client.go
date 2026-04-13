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

type BinanceClient struct {
	BaseURL    string
	AuthToken  string
	AccountID  string
	BotID      string
	SessionID  string
	HTTPClient *http.Client
}

func NewBinanceClient() *BinanceClient {
	baseURL := os.Getenv("BINANCE_TRADING_SERVER_URL")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8060"
	}
	return &BinanceClient{
		BaseURL:   baseURL,
		AuthToken: os.Getenv("BINANCE_TRADING_SERVER_AUTH_TOKEN"),
		AccountID: os.Getenv("BINANCE_ACCOUNT_ID"),
		BotID:     os.Getenv("BINANCE_BOT_ID"),
		SessionID: fmt.Sprintf("go-%d", time.Now().UnixNano()),
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (c *BinanceClient) doRequest(method, path string, body interface{}) ([]byte, error) {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(data)
	}

	url := c.BaseURL + "/api/v1" + path
	req, err := http.NewRequest(method, url, bodyReader)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.AuthToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.AuthToken)
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
	}
	return respBody, nil
}

// WriterLeaseRequest claims exclusive writer access.
type WriterLeaseRequest struct {
	Account   string `json:"account"`
	BotID     string `json:"bot_id"`
	SessionID string `json:"session_id"`
	TTL       int    `json:"ttl_seconds"`
}

type WriterLeaseResponse struct {
	OK        bool   `json:"ok"`
	TTL       int    `json:"ttl_seconds"`
	ExpiresAt string `json:"expires_at"`
}

func (c *BinanceClient) ClaimWriter(ttl int) (*WriterLeaseResponse, error) {
	if ttl == 0 {
		ttl = 1800
	}
	body, err := c.doRequest("POST", "/writer/claim", WriterLeaseRequest{
		Account:   c.AccountID,
		BotID:     c.BotID,
		SessionID: c.SessionID,
		TTL:       ttl,
	})
	if err != nil {
		return nil, fmt.Errorf("claim writer: %w", err)
	}
	var resp WriterLeaseResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

func (c *BinanceClient) HeartbeatWriter(ttl int) error {
	if ttl == 0 {
		ttl = 45
	}
	_, err := c.doRequest("POST", "/writer/heartbeat", WriterLeaseRequest{
		Account:   c.AccountID,
		BotID:     c.BotID,
		SessionID: c.SessionID,
		TTL:       ttl,
	})
	return err
}

// OrderRequest matches the server's Pydantic model.
type OrderRequest struct {
	Account        string            `json:"account"`
	BotID          string            `json:"bot_id"`
	SessionID      string            `json:"session_id"`
	Symbol         string            `json:"symbol"`
	Side           string            `json:"side"`
	Qty            float64           `json:"qty"`
	LimitPrice     float64           `json:"limit_price"`
	ExecutionMode  string            `json:"execution_mode"`
	AllowLossExit  bool              `json:"allow_loss_exit"`
	ForceExitReason *string          `json:"force_exit_reason,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

type OrderInfo struct {
	ID            string  `json:"id"`
	Account       string  `json:"account"`
	Symbol        string  `json:"symbol"`
	Side          string  `json:"side"`
	Qty           float64 `json:"qty"`
	LimitPrice    float64 `json:"limit_price"`
	Status        string  `json:"status"`
	ExecutionMode string  `json:"execution_mode"`
	CreatedAt     string  `json:"created_at"`
	FilledAt      *string `json:"filled_at"`
	FillPrice     *float64 `json:"fill_price"`
	FeeBps        int     `json:"fee_bps"`
}

type QuotePayload struct {
	Symbol   string  `json:"symbol"`
	BidPrice float64 `json:"bid_price"`
	AskPrice float64 `json:"ask_price"`
	LastPrice float64 `json:"last_price"`
	AsOf     string  `json:"as_of"`
}

type OrderResponse struct {
	Order  OrderInfo    `json:"order"`
	Quote  QuotePayload `json:"quote"`
	Filled bool         `json:"filled"`
}

func (c *BinanceClient) SubmitOrder(symbol, side string, qty, limitPrice float64, meta map[string]interface{}) (*OrderResponse, error) {
	mode := "paper"
	if os.Getenv("BINANCE_LIVE") == "1" {
		mode = "live"
	}
	body, err := c.doRequest("POST", "/orders", OrderRequest{
		Account:       c.AccountID,
		BotID:         c.BotID,
		SessionID:     c.SessionID,
		Symbol:        symbol,
		Side:          side,
		Qty:           qty,
		LimitPrice:    limitPrice,
		ExecutionMode: mode,
		Metadata:      meta,
	})
	if err != nil {
		return nil, fmt.Errorf("submit order: %w", err)
	}
	var resp OrderResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

type PositionState struct {
	Symbol        string  `json:"symbol"`
	Qty           float64 `json:"qty"`
	AvgEntryPrice float64 `json:"avg_entry_price"`
	RealizedPnl   float64 `json:"realized_pnl"`
	FeesPaid      float64 `json:"fees_paid"`
}

type AccountSnapshot struct {
	Account     string                   `json:"account"`
	Mode        string                   `json:"mode"`
	Cash        float64                  `json:"cash"`
	RealizedPnl float64                  `json:"realized_pnl"`
	TotalFees   float64                  `json:"total_fees"`
	Positions   map[string]PositionState `json:"positions"`
	UpdatedAt   string                   `json:"updated_at"`
}

func (c *BinanceClient) GetAccount() (*AccountSnapshot, error) {
	body, err := c.doRequest("GET", fmt.Sprintf("/account/%s", c.AccountID), nil)
	if err != nil {
		return nil, err
	}
	var snap AccountSnapshot
	if err := json.Unmarshal(body, &snap); err != nil {
		return nil, err
	}
	return &snap, nil
}

type RefreshPricesRequest struct {
	Account string   `json:"account"`
	Symbols []string `json:"symbols"`
}

type RefreshPricesResponse struct {
	Accounts []struct {
		Account string                  `json:"account"`
		Prices  map[string]QuotePayload `json:"prices"`
	} `json:"accounts"`
}

func (c *BinanceClient) RefreshPrices(symbols []string) (map[string]QuotePayload, error) {
	body, err := c.doRequest("POST", "/prices/refresh", RefreshPricesRequest{
		Account: c.AccountID,
		Symbols: symbols,
	})
	if err != nil {
		return nil, err
	}
	var resp RefreshPricesResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	for _, a := range resp.Accounts {
		if a.Account == c.AccountID {
			return a.Prices, nil
		}
	}
	return nil, fmt.Errorf("account %s not in response", c.AccountID)
}
