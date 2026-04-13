package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"binancego/trade"
)

func main() {
	cmd := "help"
	if len(os.Args) > 1 {
		cmd = os.Args[1]
	}

	switch cmd {
	case "trade":
		runTrade()
	case "status":
		runStatus()
	case "prices":
		runPrices()
	default:
		fmt.Println("binancego tradecli -- live trading with ONNX models")
		fmt.Println()
		fmt.Println("Commands:")
		fmt.Println("  trade    Start live/paper trading loop")
		fmt.Println("  status   Show account snapshot")
		fmt.Println("  prices   Refresh and show prices")
		fmt.Println()
		fmt.Println("Env vars:")
		fmt.Println("  BINANCE_TRADING_SERVER_URL          (default http://127.0.0.1:8060)")
		fmt.Println("  BINANCE_TRADING_SERVER_AUTH_TOKEN    auth token")
		fmt.Println("  BINANCE_ACCOUNT_ID                  account name")
		fmt.Println("  BINANCE_BOT_ID                      bot identifier")
		fmt.Println("  BINANCE_LIVE=1                      enable live trading (default paper)")
	}
}

func runTrade() {
	fs := flag.NewFlagSet("trade", flag.ExitOnError)
	modelPath := fs.String("model", "", "path to .onnx model")
	forecastDir := fs.String("forecasts", "", "path to forecast cache dir")
	metaPath := fs.String("meta", "", "path to training_meta.json")
	symbols := fs.String("symbols", "BTCUSD,ETHUSD,SOLUSD,DOGEUSD,AAVEUSD,LINKUSD", "comma-separated symbols")
	seqLen := fs.Int("seq-len", 48, "model sequence length")
	dataRoot := fs.String("data-root", "trainingdatahourly/crypto", "OHLCV data root")
	stateDir := fs.String("state-dir", "/tmp/binancego_state", "state directory for locks/buys")
	interval := fs.Duration("interval", 1*time.Hour, "trading interval")
	fs.Parse(os.Args[2:])

	if *modelPath == "" {
		log.Fatal("--model required")
	}

	cfg := trade.TradingLoopConfig{
		StateDir:    *stateDir,
		ModelPath:   *modelPath,
		ForecastDir: *forecastDir,
		MetaPath:    *metaPath,
		Symbols:     strings.Split(*symbols, ","),
		SeqLen:      *seqLen,
		DataRoot:    *dataRoot,
		Interval:    *interval,
	}

	loop, err := trade.NewTradingLoop(cfg)
	if err != nil {
		log.Fatalf("init: %v", err)
	}
	loop.Run()
}

func runStatus() {
	client := trade.NewBinanceClient()
	acct, err := client.GetAccount()
	if err != nil {
		log.Fatalf("get account: %v", err)
	}

	fmt.Printf("Account: %s (%s)\n", acct.Account, acct.Mode)
	fmt.Printf("Cash: %.2f  PnL: %.2f  Fees: %.2f\n", acct.Cash, acct.RealizedPnl, acct.TotalFees)
	if len(acct.Positions) > 0 {
		fmt.Printf("\nPositions:\n")
		for sym, pos := range acct.Positions {
			fmt.Printf("  %-10s qty=%.6f entry=%.2f pnl=%.2f\n",
				sym, pos.Qty, pos.AvgEntryPrice, pos.RealizedPnl)
		}
	}
}

func runPrices() {
	fs := flag.NewFlagSet("prices", flag.ExitOnError)
	symbols := fs.String("symbols", "BTCUSD,ETHUSD,SOLUSD", "symbols")
	fs.Parse(os.Args[2:])

	client := trade.NewBinanceClient()
	syms := strings.Split(*symbols, ",")
	prices, err := client.RefreshPrices(syms)
	if err != nil {
		log.Fatalf("refresh prices: %v", err)
	}

	for sym, q := range prices {
		fmt.Printf("%-10s bid=%.2f ask=%.2f last=%.2f\n",
			sym, q.BidPrice, q.AskPrice, q.LastPrice)
	}
}
