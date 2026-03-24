#pragma once
#include <string>
#include <vector>
#include "policy.h"
#include "obs_norm.h"

struct CheckpointMeta {
    int update = 0;
    int global_step = 0;
    float best_return = -1e9f;
    float best_sortino = -1e9f;
    int hidden_size = 1024;
    int obs_size = 0;
    int num_actions = 0;
};

void save_checkpoint(const std::string& path, TradingPolicy& policy,
                     const RunningObsNorm& obs_norm, const CheckpointMeta& meta);

bool load_checkpoint(const std::string& path, TradingPolicy& policy,
                     RunningObsNorm& obs_norm, CheckpointMeta& meta);

struct TopKEntry {
    std::string path;
    float metric;
};

class TopKManager {
public:
    TopKManager(int k = 5) : k_(k) {}
    void add(const std::string& path, float metric);
    const std::vector<TopKEntry>& entries() const { return entries_; }
    void save_manifest(const std::string& dir) const;

private:
    int k_;
    std::vector<TopKEntry> entries_;
};
