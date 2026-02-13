package main

import (
	"fmt"

	"github.com/alpacahq/alpaca-trade-api-go/v3/marketdata"
)

// QuoteResult is a unified bid/ask for both crypto and stocks.
type QuoteResult struct {
	Symbol   string
	BidPrice float64
	AskPrice float64
	MidPrice float64
}

// GetLatestQuote fetches the current bid/ask for a symbol.
func (t *Trader) GetLatestQuote(symbol string) (*QuoteResult, error) {
	sym := remapSymbol(symbol)
	if isCrypto(symbol) {
		return t.getCryptoQuote(sym)
	}
	return t.getStockQuote(sym)
}

func (t *Trader) getCryptoQuote(symbol string) (*QuoteResult, error) {
	mdClient := marketdata.NewClient(marketdata.ClientOpts{
		APIKey:    t.cfg.APIKeyID,
		APISecret: t.cfg.SecretKey,
	})
	q, err := mdClient.GetLatestCryptoQuote(symbol, marketdata.GetLatestCryptoQuoteRequest{})
	if err != nil {
		return nil, fmt.Errorf("crypto quote %s: %w", symbol, err)
	}
	mid := (q.BidPrice + q.AskPrice) / 2.0
	return &QuoteResult{
		Symbol:   symbol,
		BidPrice: q.BidPrice,
		AskPrice: q.AskPrice,
		MidPrice: mid,
	}, nil
}

func (t *Trader) getStockQuote(symbol string) (*QuoteResult, error) {
	mdClient := marketdata.NewClient(marketdata.ClientOpts{
		APIKey:    t.cfg.APIKeyID,
		APISecret: t.cfg.SecretKey,
	})
	q, err := mdClient.GetLatestQuote(symbol, marketdata.GetLatestQuoteRequest{})
	if err != nil {
		return nil, fmt.Errorf("stock quote %s: %w", symbol, err)
	}
	mid := (q.BidPrice + q.AskPrice) / 2.0
	return &QuoteResult{
		Symbol:   symbol,
		BidPrice: q.BidPrice,
		AskPrice: q.AskPrice,
		MidPrice: mid,
	}, nil
}

// SpreadPct returns the bid-ask spread as a percentage of midpoint.
func (q *QuoteResult) SpreadPct() float64 {
	if q.MidPrice == 0 {
		return 0
	}
	return (q.AskPrice - q.BidPrice) / q.MidPrice
}
