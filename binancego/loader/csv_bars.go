// Package loader reads OHLCV CSV bar files (Alpaca hourly/daily exports)
// into mixed.Candle slices aligned across multiple symbols. Kept separate
// from binancego/data so the broken parquet/feature stubs there do not
// transitively break the mixed-policy backtest path.
package loader

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"binancego/policy/mixed"
)

// Hourly stock bar files in /nvme0n1-disk/code/stock-prediction/trainingdatahourly/stocks
// have header: timestamp,open,high,low,close,volume,trade_count,vwap,symbol

type CSVBarRow struct {
	Timestamp time.Time
	Symbol    string
	Open      float64
	High      float64
	Low       float64
	Close     float64
	Volume    float64
}

// LoadStockCSVBars loads an Alpaca hourly bar CSV from the
// trainingdatahourly/stocks/ tree and returns rows in chronological order.
// Lines that fail to parse are skipped silently (corrupt rows occasionally
// occur after splits/halts).
func LoadStockCSVBars(path string) ([]CSVBarRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var header []string
	rows := make([]CSVBarRow, 0, 4096)
	first := true
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		if first {
			header = fields
			first = false
			continue
		}
		// Tolerate variable column orders by name index.
		row, ok := parseStockCSVRow(fields, header)
		if !ok {
			continue
		}
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan %s: %w", path, err)
	}
	sort.SliceStable(rows, func(i, j int) bool { return rows[i].Timestamp.Before(rows[j].Timestamp) })
	return rows, nil
}

func parseStockCSVRow(fields, header []string) (CSVBarRow, bool) {
	if len(fields) < len(header) {
		return CSVBarRow{}, false
	}
	col := func(name string) string {
		for i, h := range header {
			if h == name {
				return fields[i]
			}
		}
		return ""
	}
	tsStr := col("timestamp")
	if tsStr == "" {
		return CSVBarRow{}, false
	}
	ts, err := parseAlpacaTimestamp(tsStr)
	if err != nil {
		return CSVBarRow{}, false
	}
	open, errO := strconv.ParseFloat(col("open"), 64)
	high, errH := strconv.ParseFloat(col("high"), 64)
	low, errL := strconv.ParseFloat(col("low"), 64)
	closeP, errC := strconv.ParseFloat(col("close"), 64)
	volume, _ := strconv.ParseFloat(col("volume"), 64)
	if errO != nil || errH != nil || errL != nil || errC != nil {
		return CSVBarRow{}, false
	}
	if open <= 0 || high <= 0 || low <= 0 || closeP <= 0 {
		return CSVBarRow{}, false
	}
	return CSVBarRow{
		Timestamp: ts,
		Symbol:    col("symbol"),
		Open:      open,
		High:      high,
		Low:       low,
		Close:     closeP,
		Volume:    volume,
	}, true
}

// parseAlpacaTimestamp handles "2024-02-20 14:30:00+00:00" plus a few common
// fallbacks (RFC3339, ISO without tz).
func parseAlpacaTimestamp(s string) (time.Time, error) {
	// Most common Alpaca format: "YYYY-MM-DD HH:MM:SS+00:00"
	layouts := []string{
		"2006-01-02 15:04:05-07:00",
		"2006-01-02 15:04:05Z07:00",
		"2006-01-02 15:04:05+00:00",
		"2006-01-02T15:04:05Z07:00",
		time.RFC3339,
	}
	for _, layout := range layouts {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("unrecognised timestamp %q", s)
}

// LoadStockCandles reads a single CSV file and returns []mixed.Candle.
// Timestamp is preserved as UnixNano so downstream consumers can format it.
func LoadStockCandles(path string) ([]mixed.Candle, error) {
	rows, err := LoadStockCSVBars(path)
	if err != nil {
		return nil, err
	}
	out := make([]mixed.Candle, 0, len(rows))
	for _, r := range rows {
		out = append(out, mixed.Candle{
			Timestamp: r.Timestamp.UnixNano(),
			Open:      r.Open,
			High:      r.High,
			Low:       r.Low,
			Close:     r.Close,
			Volume:    r.Volume,
		})
	}
	return out, nil
}

// LoadAlignedStockCandles loads multiple symbols and aligns them onto a
// shared timestamp grid. The grid starts at the LATEST first-bar timestamp
// across all loaded symbols (so every symbol has real data from bar 0) and
// is the union of timestamps thereafter. Missing intermediate bars are
// filled by carrying forward the previous Close (zero-volume), giving
// identical length + identical timestamp at each index. This is what
// mixed.SimulatePortfolio expects.
//
// rootDir defaults to trainingdatahourly/stocks when empty. Pass full symbol
// names (e.g. "AAPL") — the function appends ".csv".
//
// Symbols that fail to load are silently dropped; the returned slice contains
// only successfully aligned series.
func LoadAlignedStockCandles(rootDir string, symbols []string) ([]mixed.SymbolBars, error) {
	if rootDir == "" {
		rootDir = "trainingdatahourly/stocks"
	}
	type loaded struct {
		Symbol  string
		Candles []mixed.Candle
	}
	all := make([]loaded, 0, len(symbols))
	for _, sym := range symbols {
		path := filepath.Join(rootDir, sym+".csv")
		c, err := LoadStockCandles(path)
		if err != nil || len(c) == 0 {
			continue
		}
		all = append(all, loaded{Symbol: sym, Candles: c})
	}
	if len(all) == 0 {
		return nil, fmt.Errorf("no symbols loaded from %q", rootDir)
	}
	// Master start = latest first-bar across all loaded symbols.
	masterStart := all[0].Candles[0].Timestamp
	for _, l := range all[1:] {
		if l.Candles[0].Timestamp > masterStart {
			masterStart = l.Candles[0].Timestamp
		}
	}
	// Build union of timestamps >= masterStart.
	tsSet := map[int64]struct{}{}
	for _, l := range all {
		for _, b := range l.Candles {
			if b.Timestamp >= masterStart {
				tsSet[b.Timestamp] = struct{}{}
			}
		}
	}
	tsList := make([]int64, 0, len(tsSet))
	for ts := range tsSet {
		tsList = append(tsList, ts)
	}
	sort.Slice(tsList, func(i, j int) bool { return tsList[i] < tsList[j] })

	out := make([]mixed.SymbolBars, 0, len(all))
	for _, l := range all {
		idx := make(map[int64]mixed.Candle, len(l.Candles))
		for _, c := range l.Candles {
			idx[c.Timestamp] = c
		}
		aligned := make([]mixed.Candle, 0, len(tsList))
		var lastClose float64
		for _, ts := range tsList {
			if c, ok := idx[ts]; ok {
				aligned = append(aligned, c)
				lastClose = c.Close
				continue
			}
			// All symbols have data from masterStart, so any gap mid-series gets
			// forward-filled (zero volume to suppress strategy entries).
			aligned = append(aligned, mixed.Candle{
				Timestamp: ts,
				Open:      lastClose,
				High:      lastClose,
				Low:       lastClose,
				Close:     lastClose,
				Volume:    0,
			})
		}
		out = append(out, mixed.SymbolBars{Symbol: l.Symbol, Bars: aligned})
	}
	return out, nil
}
