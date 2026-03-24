#pragma once
#include <torch/torch.h>
#include <tuple>

struct TradingPolicyImpl : torch::nn::Module {
    torch::nn::Sequential encoder{nullptr};
    torch::nn::Sequential actor{nullptr};
    torch::nn::Sequential critic{nullptr};
    int obs_size_;
    int num_actions_;
    int hidden_;

    TradingPolicyImpl(int obs_size, int num_actions, int hidden = 1024);

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    get_action_and_value(torch::Tensor x, torch::Tensor action = {});

    torch::Tensor get_value(torch::Tensor x);
};

TORCH_MODULE(TradingPolicy);
