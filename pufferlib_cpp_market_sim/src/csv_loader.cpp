#include "csv_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace market_sim {

CSVData CSVLoader::load(const std::string& filepath) {
    CSVData data;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filepath);
    }

    std::string line;
    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        auto fields = split_line(line, ',');
        if (fields.size() < 6) {
            throw std::runtime_error("Invalid CSV format: expected at least 6 fields, got " +
                                   std::to_string(fields.size()));
        }

        // Parse fields: timestamp,open,high,low,close,volume (ignore extra fields if present)
        // Try to parse timestamp as int64, if that fails try as string (for datetime format)
        try {
            data.timestamps.push_back(std::stoll(fields[0]));
        } catch (const std::exception&) {
            // If timestamp is datetime string, use hash or index
            data.timestamps.push_back(static_cast<int64_t>(data.timestamps.size()));
        }
        data.open.push_back(std::stof(fields[1]));
        data.high.push_back(std::stof(fields[2]));
        data.low.push_back(std::stof(fields[3]));
        data.close.push_back(std::stof(fields[4]));
        data.volume.push_back(std::stof(fields[5]));
    }

    file.close();
    return data;
}

torch::Tensor CSVLoader::to_tensor(
    const std::vector<float>& data,
    torch::Device device
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto tensor = torch::from_blob(
        const_cast<float*>(data.data()),
        {static_cast<int64_t>(data.size())},
        options
    ).clone();  // Clone to own the data
    return tensor.to(device);
}

torch::Tensor CSVLoader::to_tensor_int64(
    const std::vector<int64_t>& data,
    torch::Device device
) {
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    auto tensor = torch::from_blob(
        const_cast<int64_t*>(data.data()),
        {static_cast<int64_t>(data.size())},
        options
    ).clone();
    return tensor.to(device);
}

bool CSVLoader::is_crypto_symbol(const std::string& symbol) {
    // Common crypto patterns: ends with USD, BTC, ETH, etc.
    std::string upper_symbol = symbol;
    std::transform(upper_symbol.begin(), upper_symbol.end(),
                   upper_symbol.begin(), ::toupper);

    return upper_symbol.find("USD") != std::string::npos ||
           upper_symbol.find("BTC") != std::string::npos ||
           upper_symbol.find("ETH") != std::string::npos ||
           upper_symbol.find("USDT") != std::string::npos;
}

std::vector<std::string> CSVLoader::split_line(
    const std::string& line,
    char delimiter
) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;

    while (std::getline(ss, field, delimiter)) {
        fields.push_back(field);
    }

    return fields;
}

} // namespace market_sim
