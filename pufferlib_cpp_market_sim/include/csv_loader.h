#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>

namespace market_sim {

struct CSVData {
    std::vector<int64_t> timestamps;
    std::vector<float> open;
    std::vector<float> high;
    std::vector<float> low;
    std::vector<float> close;
    std::vector<float> volume;
};

class CSVLoader {
public:
    // Load CSV file from trainingdata/
    // Expected format: timestamp,open,high,low,close,volume
    static CSVData load(const std::string& filepath);

    // Convert CSVData to GPU tensors
    static torch::Tensor to_tensor(
        const std::vector<float>& data,
        torch::Device device
    );

    static torch::Tensor to_tensor_int64(
        const std::vector<int64_t>& data,
        torch::Device device
    );

    // Detect if symbol is crypto based on name
    static bool is_crypto_symbol(const std::string& symbol);

private:
    static std::vector<std::string> split_line(const std::string& line, char delimiter);
};

} // namespace market_sim
