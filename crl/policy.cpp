#include "policy.h"
#include <cmath>

static void orthogonal_init(torch::nn::Module& mod, double gain) {
    for (auto& p : mod.named_parameters()) {
        if (p.key().find("weight") != std::string::npos && p.value().dim() >= 2) {
            torch::nn::init::orthogonal_(p.value(), gain);
        }
        if (p.key().find("bias") != std::string::npos) {
            torch::nn::init::zeros_(p.value());
        }
    }
}

TradingPolicyImpl::TradingPolicyImpl(int obs_size, int num_actions, int hidden)
    : obs_size_(obs_size), num_actions_(num_actions), hidden_(hidden) {

    encoder = register_module("encoder", torch::nn::Sequential(
        torch::nn::Linear(obs_size, hidden),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden, hidden),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden, hidden),
        torch::nn::ReLU()
    ));

    actor = register_module("actor", torch::nn::Sequential(
        torch::nn::Linear(hidden, hidden / 2),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden / 2, num_actions)
    ));

    critic = register_module("critic", torch::nn::Sequential(
        torch::nn::Linear(hidden, hidden / 2),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden / 2, 1)
    ));

    orthogonal_init(*encoder, std::sqrt(2.0));
    orthogonal_init(*actor, std::sqrt(2.0));
    orthogonal_init(*critic, std::sqrt(2.0));

    // actor last layer: gain=0.01
    auto actor_params = actor->parameters();
    if (actor_params.size() >= 2) {
        torch::nn::init::orthogonal_(actor_params[actor_params.size() - 2], 0.01);
        torch::nn::init::zeros_(actor_params[actor_params.size() - 1]);
    }
    // critic last layer: gain=1.0
    auto critic_params = critic->parameters();
    if (critic_params.size() >= 2) {
        torch::nn::init::orthogonal_(critic_params[critic_params.size() - 2], 1.0);
        torch::nn::init::zeros_(critic_params[critic_params.size() - 1]);
    }
}

std::pair<torch::Tensor, torch::Tensor> TradingPolicyImpl::forward(torch::Tensor x) {
    auto h = encoder->forward(x);
    auto logits = actor->forward(h);
    auto value = critic->forward(h).squeeze(-1);
    return {logits, value};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TradingPolicyImpl::get_action_and_value(torch::Tensor x, torch::Tensor action) {
    auto [logits, value] = forward(x);

    auto log_probs = torch::log_softmax(logits, -1);
    auto probs = torch::softmax(logits, -1);
    auto entropy = -(probs * log_probs).sum(-1);

    if (!action.defined() || action.numel() == 0) {
        action = torch::multinomial(probs, 1).squeeze(-1);
    }

    auto log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1);
    return {action, log_prob, entropy, value};
}

torch::Tensor TradingPolicyImpl::get_value(torch::Tensor x) {
    auto h = encoder->forward(x);
    return critic->forward(h).squeeze(-1);
}
