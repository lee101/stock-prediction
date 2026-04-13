package policy

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

var ortInitialized bool

func InitONNXRuntime(libPath string) error {
	if ortInitialized {
		return nil
	}
	if libPath == "" {
		for _, p := range []string{
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/libonnxruntime.so",
			"libonnxruntime.so",
		} {
			if _, err := os.Stat(p); err == nil {
				libPath = p
				break
			}
		}
	}
	if libPath == "" {
		return fmt.Errorf("onnxruntime shared library not found")
	}
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("init onnxruntime: %w", err)
	}
	ortInitialized = true
	return nil
}

func DestroyONNXRuntime() {
	if ortInitialized {
		ort.DestroyEnvironment()
		ortInitialized = false
	}
}

type ONNXModel struct {
	ModelPath  string
	session    *ort.DynamicAdvancedSession
	InputDim   int
	SeqLen     int
	NumOutputs int
}

func LoadONNX(path string) (*ONNXModel, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, fmt.Errorf("model not found: %s", path)
	}
	if !ortInitialized {
		if err := InitONNXRuntime(""); err != nil {
			return nil, err
		}
	}

	session, err := ort.NewDynamicAdvancedSession(path,
		[]string{"features"}, []string{"logits"}, nil)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}

	return &ONNXModel{
		ModelPath: path,
		session:   session,
	}, nil
}

func (m *ONNXModel) Close() {
	if m.session != nil {
		m.session.Destroy()
	}
}

// Infer runs inference on features [batch, seq_len, input_dim].
// Returns logits flattened and num_outputs.
func (m *ONNXModel) Infer(features []float32, batch, seqLen, inputDim int) ([]float32, int, error) {
	inputShape := ort.Shape{int64(batch), int64(seqLen), int64(inputDim)}
	inputTensor, err := ort.NewTensor(inputShape, features)
	if err != nil {
		return nil, 0, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Pass nil output for auto-allocation
	outputs := []ort.Value{nil}
	err = m.session.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return nil, 0, fmt.Errorf("run inference: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, 0, fmt.Errorf("unexpected output type")
	}

	shape := outTensor.GetShape()
	numOutputs := 4
	if len(shape) >= 3 {
		numOutputs = int(shape[2])
	}

	data := outTensor.GetData()
	result := make([]float32, len(data))
	copy(result, data)

	return result, numOutputs, nil
}

// InferSequence runs single-batch inference on [seq_len][input_dim] features.
func (m *ONNXModel) InferSequence(features [][]float32) ([][]float64, error) {
	if len(features) == 0 {
		return nil, fmt.Errorf("empty features")
	}
	seqLen := len(features)
	inputDim := len(features[0])

	flat := make([]float32, seqLen*inputDim)
	for i, row := range features {
		copy(flat[i*inputDim:], row)
	}

	outFlat, numOutputs, err := m.Infer(flat, 1, seqLen, inputDim)
	if err != nil {
		return nil, err
	}

	result := make([][]float64, seqLen)
	for i := 0; i < seqLen; i++ {
		result[i] = make([]float64, numOutputs)
		for j := 0; j < numOutputs; j++ {
			idx := i*numOutputs + j
			if idx < len(outFlat) {
				result[i][j] = float64(outFlat[idx])
			}
		}
	}
	return result, nil
}

func ExportCheckpointToONNX(checkpointPath string, seqLen int) (string, error) {
	outputPath := checkpointPath[:len(checkpointPath)-3] + ".onnx"
	if _, err := os.Stat(outputPath); err == nil {
		return outputPath, nil
	}

	scriptPath := filepath.Join(filepath.Dir(os.Args[0]), "..", "policy", "export_onnx.py")
	if _, err := os.Stat(scriptPath); err != nil {
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
