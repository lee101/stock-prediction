package policy

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// ONNXModel wraps an ONNX model for inference.
// NOTE: Full ONNX Runtime integration requires github.com/yalue/onnxruntime_go
// which needs the onnxruntime shared library installed.
// This file provides the interface and a fallback using Python subprocess.
type ONNXModel struct {
	ModelPath string
	InputDim  int
	SeqLen    int
	NumOutputs int
}

// LoadONNX loads an ONNX model file.
func LoadONNX(path string) (*ONNXModel, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("model not found: %s", path)
	}
	return &ONNXModel{ModelPath: path}, nil
}

// ExportCheckpointToONNX calls the Python export script.
func ExportCheckpointToONNX(checkpointPath string, seqLen int) (string, error) {
	outputPath := checkpointPath[:len(checkpointPath)-3] + ".onnx"
	if _, err := os.Stat(outputPath); err == nil {
		return outputPath, nil // already exported
	}

	scriptPath := filepath.Join(filepath.Dir(os.Args[0]), "..", "policy", "export_onnx.py")
	// Try relative to binary, then absolute path
	if _, err := os.Stat(scriptPath); err != nil {
		// Try from binancego root
		scriptPath = "binancego/policy/export_onnx.py"
	}

	cmd := exec.Command("python", scriptPath,
		checkpointPath,
		"--output", outputPath,
		"--seq-len", fmt.Sprintf("%d", seqLen),
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("export failed: %w", err)
	}
	return outputPath, nil
}

// InferPython runs inference via Python subprocess (fallback when ONNX Runtime not available).
// Returns logits as [][]float64 (seq_len x num_outputs).
func InferPython(checkpointPath string, features [][]float32) ([][]float64, error) {
	// TODO: implement via Python subprocess with JSON I/O
	return nil, fmt.Errorf("InferPython not yet implemented; install onnxruntime_go for native inference")
}
