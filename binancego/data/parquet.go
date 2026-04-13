package data

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/parquet-go/parquet-go"
)

type ForecastRow struct {
	Timestamp         int64   `parquet:"timestamp,timestamp(millisecond)"`
	Symbol            string  `parquet:"symbol"`
	IssuedAt          int64   `parquet:"issued_at,timestamp(millisecond)"`
	TargetTimestamp   int64   `parquet:"target_timestamp,timestamp(millisecond)"`
	HorizonHours      int64   `parquet:"horizon_hours"`
	PredictedCloseP50 float64 `parquet:"predicted_close_p50"`
	PredictedCloseP10 float64 `parquet:"predicted_close_p10"`
	PredictedCloseP90 float64 `parquet:"predicted_close_p90"`
	PredictedHighP50  float64 `parquet:"predicted_high_p50"`
	PredictedLowP50   float64 `parquet:"predicted_low_p50"`
}

func (r ForecastRow) Time() time.Time {
	return time.Unix(0, r.Timestamp*int64(time.Millisecond)).UTC()
}

type ForecastCache struct {
	Symbol      string
	Rows        []ForecastRow
	byTimestamp map[int64]int // unix seconds -> index
}

func LoadForecastParquet(path string) (*ForecastCache, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	reader := parquet.NewGenericReader[ForecastRow](f)
	_ = stat

	rows := make([]ForecastRow, 0, 4096)
	buf := make([]ForecastRow, 256)
	for {
		n, err := reader.Read(buf)
		if n > 0 {
			rows = append(rows, buf[:n]...)
		}
		if err != nil {
			break
		}
	}
	reader.Close()

	if len(rows) == 0 {
		return nil, fmt.Errorf("no rows in %s", path)
	}

	sort.Slice(rows, func(i, j int) bool {
		return rows[i].Timestamp < rows[j].Timestamp
	})

	byTS := make(map[int64]int, len(rows))
	for i, r := range rows {
		// Convert milliseconds to seconds for lookup
		tsSec := r.Timestamp / 1000
		byTS[tsSec] = i
	}

	return &ForecastCache{
		Symbol:      rows[0].Symbol,
		Rows:        rows,
		byTimestamp: byTS,
	}, nil
}

func (fc *ForecastCache) GetForecast(unixTS int64) (ForecastRow, bool) {
	idx, ok := fc.byTimestamp[unixTS]
	if !ok {
		return ForecastRow{}, false
	}
	return fc.Rows[idx], true
}

func (fc *ForecastCache) GetForecasts(bars []TimestampedBar) []ForecastRow {
	result := make([]ForecastRow, len(bars))
	for i, b := range bars {
		ts := b.Timestamp.Unix()
		if idx, ok := fc.byTimestamp[ts]; ok {
			result[i] = fc.Rows[idx]
		}
	}
	return result
}

func LoadForecastDir(dir string) (map[string]*ForecastCache, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.parquet"))
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no parquet files in %s", dir)
	}

	caches := make(map[string]*ForecastCache)
	for _, f := range files {
		base := filepath.Base(f)
		symbol := base[:len(base)-len(".parquet")]
		cache, err := LoadForecastParquet(f)
		if err != nil {
			fmt.Printf("  %s: skip (%v)\n", symbol, err)
			continue
		}
		caches[symbol] = cache
	}
	return caches, nil
}
