package policy

import (
	"os"
	"testing"
)

func TestONNXIntegration(t *testing.T) {
	modelPath := "/tmp/btc_epoch_001.onnx"
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("ONNX model not found at %s (run export_onnx.py first)", modelPath)
	}

	// Use newer onnxruntime if available (API v20+)
	ortLib := os.Getenv("ORT_LIB_PATH")
	if ortLib == "" {
		ortLib = "/tmp/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so"
	}
	if _, err := os.Stat(ortLib); err != nil {
		t.Skipf("onnxruntime library not found at %s", ortLib)
	}
	if err := InitONNXRuntime(ortLib); err != nil {
		t.Fatalf("InitONNXRuntime: %v", err)
	}
	defer DestroyONNXRuntime()

	model, err := LoadONNX(modelPath)
	if err != nil {
		t.Fatalf("LoadONNX: %v", err)
	}
	defer model.Close()

	// input_dim=16, seq_len=48
	seqLen := 48
	inputDim := 16
	features := make([][]float32, seqLen)
	for i := range features {
		row := make([]float32, inputDim)
		for j := range row {
			row[j] = float32(i) * 0.01
		}
		features[i] = row
	}

	logits, err := model.InferSequence(features)
	if err != nil {
		t.Fatalf("InferSequence: %v", err)
	}

	if len(logits) != seqLen {
		t.Fatalf("expected %d timesteps, got %d", seqLen, len(logits))
	}

	// Should have 4 outputs (buy_price, sell_price, buy_amount, sell_amount)
	if len(logits[0]) < 4 {
		t.Fatalf("expected >=4 outputs per timestep, got %d", len(logits[0]))
	}

	t.Logf("output shape: [%d, %d]", len(logits), len(logits[0]))
	t.Logf("last logits: %v", logits[seqLen-1])

	// Decode actions from last timestep
	lastLogits := logits[seqLen-1]
	cfg := DefaultDecodeConfig()
	action := DecodeActions(lastLogits, 50000.0, 50500.0, 49500.0, cfg)

	t.Logf("decoded action: buy=%.2f sell=%.2f buyAmt=%.1f sellAmt=%.1f",
		action.BuyPrice, action.SellPrice, action.BuyAmount, action.SellAmount)

	if action.BuyPrice <= 0 || action.SellPrice <= 0 {
		t.Error("decoded prices should be positive")
	}
	if action.SellPrice <= action.BuyPrice {
		t.Error("sell should be above buy")
	}
}
