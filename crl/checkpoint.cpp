#include "checkpoint.h"
#include "vendor/cJSON.h"
#include <fstream>
#include <algorithm>
#include <cstdio>

void save_checkpoint(const std::string& path, TradingPolicy& policy,
                     const RunningObsNorm& obs_norm, const CheckpointMeta& meta) {
    torch::save(policy, path);

    // save metadata as JSON sidecar
    std::string json_path = path + ".meta.json";
    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "update", meta.update);
    cJSON_AddNumberToObject(root, "global_step", meta.global_step);
    cJSON_AddNumberToObject(root, "best_return", meta.best_return);
    cJSON_AddNumberToObject(root, "best_sortino", meta.best_sortino);
    cJSON_AddNumberToObject(root, "hidden_size", meta.hidden_size);
    cJSON_AddNumberToObject(root, "obs_size", meta.obs_size);
    cJSON_AddNumberToObject(root, "num_actions", meta.num_actions);

    if (obs_norm.size_ > 0) {
        cJSON_AddNumberToObject(root, "obs_norm_size", obs_norm.size_);
        cJSON_AddNumberToObject(root, "obs_norm_count", obs_norm.count_);
        cJSON* mean_arr = cJSON_CreateArray();
        cJSON* var_arr = cJSON_CreateArray();
        for (int i = 0; i < obs_norm.size_; i++) {
            cJSON_AddItemToArray(mean_arr, cJSON_CreateNumber(obs_norm.mean_[i]));
            cJSON_AddItemToArray(var_arr, cJSON_CreateNumber(obs_norm.var_[i]));
        }
        cJSON_AddItemToObject(root, "obs_norm_mean", mean_arr);
        cJSON_AddItemToObject(root, "obs_norm_var", var_arr);
    }

    char* json_str = cJSON_Print(root);
    std::ofstream f(json_path);
    f << json_str;
    f.close();
    free(json_str);
    cJSON_Delete(root);
}

bool load_checkpoint(const std::string& path, TradingPolicy& policy,
                     RunningObsNorm& obs_norm, CheckpointMeta& meta) {
    try {
        torch::load(policy, path);
    } catch (const std::exception& e) {
        fprintf(stderr, "load_checkpoint: failed to load %s: %s\n", path.c_str(), e.what());
        return false;
    }

    std::string json_path = path + ".meta.json";
    std::ifstream f(json_path);
    if (!f.good()) return true; // model loaded but no meta

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    cJSON* root = cJSON_Parse(content.c_str());
    if (!root) return true;

    auto get_int = [&](const char* key, int def) -> int {
        cJSON* item = cJSON_GetObjectItem(root, key);
        return item ? item->valueint : def;
    };
    auto get_float = [&](const char* key, float def) -> float {
        cJSON* item = cJSON_GetObjectItem(root, key);
        return item ? (float)item->valuedouble : def;
    };

    meta.update = get_int("update", 0);
    meta.global_step = get_int("global_step", 0);
    meta.best_return = get_float("best_return", -1e9f);
    meta.best_sortino = get_float("best_sortino", -1e9f);
    meta.hidden_size = get_int("hidden_size", 1024);
    meta.obs_size = get_int("obs_size", 0);
    meta.num_actions = get_int("num_actions", 0);

    int norm_size = get_int("obs_norm_size", 0);
    if (norm_size > 0) {
        obs_norm.init(norm_size);
        obs_norm.count_ = get_float("obs_norm_count", 1e-4);
        cJSON* mean_arr = cJSON_GetObjectItem(root, "obs_norm_mean");
        cJSON* var_arr = cJSON_GetObjectItem(root, "obs_norm_var");
        if (mean_arr && var_arr) {
            for (int i = 0; i < norm_size; i++) {
                obs_norm.mean_[i] = cJSON_GetArrayItem(mean_arr, i)->valuedouble;
                obs_norm.var_[i] = cJSON_GetArrayItem(var_arr, i)->valuedouble;
            }
        }
    }

    cJSON_Delete(root);
    return true;
}

void TopKManager::add(const std::string& path, float metric) {
    entries_.push_back({path, metric});
    std::sort(entries_.begin(), entries_.end(),
              [](const TopKEntry& a, const TopKEntry& b) { return a.metric > b.metric; });
    while ((int)entries_.size() > k_) {
        auto& removed = entries_.back();
        std::remove(removed.path.c_str());
        entries_.pop_back();
    }
}

void TopKManager::save_manifest(const std::string& dir) const {
    std::string path = dir + "/topk_manifest.json";
    cJSON* root = cJSON_CreateArray();
    for (auto& e : entries_) {
        cJSON* item = cJSON_CreateObject();
        cJSON_AddStringToObject(item, "path", e.path.c_str());
        cJSON_AddNumberToObject(item, "metric", e.metric);
        cJSON_AddItemToArray(root, item);
    }
    char* json_str = cJSON_Print(root);
    std::ofstream f(path);
    f << json_str;
    f.close();
    free(json_str);
    cJSON_Delete(root);
}
