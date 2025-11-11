#!/usr/bin/env python3
"""
Fast PPO training for market simulation using CUDA-accelerated environment

This demonstrates how to use the C/CUDA market simulation with RL training.
Based on pufferlib's high-performance training architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
import time

# Try to import custom CUDA kernels
try:
    import market_sim_cuda
    CUDA_KERNELS_AVAILABLE = True
except ImportError:
    CUDA_KERNELS_AVAILABLE = False
    print("Warning: CUDA kernels not available. Using PyTorch implementations.")


class MarketPPONetwork(nn.Module):
    """
    PPO policy and value network for market trading

    Architecture optimized for financial time series:
    - Conv1D layers for temporal feature extraction
    - LSTM for sequential modeling
    - Separate policy and value heads
    """

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 512,
        lstm_hidden: int = 256,
    ):
        super().__init__()

        # Feature extraction with 1D convolutions
        self.conv1 = nn.Conv1d(1, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1)

        # Calculate conv output size (simplified, adjust based on actual obs_size)
        conv_out_size = 128 * ((obs_size - 14) // 8)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(conv_out_size, lstm_hidden, batch_first=True)

        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, obs, hidden_state=None):
        """
        Forward pass through the network

        Args:
            obs: [batch, obs_size] or [batch, seq_len, obs_size]
            hidden_state: Optional LSTM hidden state

        Returns:
            policy_logits, value, new_hidden_state
        """
        # Handle batched input
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # [batch, 1, obs_size]

        # Conv feature extraction
        x = obs.transpose(1, 2)  # [batch, obs_size, seq_len]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten for LSTM
        x = x.flatten(start_dim=1).unsqueeze(1)  # [batch, 1, features]

        # LSTM
        if hidden_state is None:
            lstm_out, new_hidden = self.lstm(x)
        else:
            lstm_out, new_hidden = self.lstm(x, hidden_state)

        lstm_out = lstm_out.squeeze(1)  # [batch, lstm_hidden]

        # Policy and value heads
        policy_logits = self.policy(lstm_out)
        value = self.value(lstm_out).squeeze(-1)

        return policy_logits, value, new_hidden


class PPOTrainer:
    """
    High-performance PPO trainer with CUDA acceleration
    """

    def __init__(
        self,
        env,
        network: MarketPPONetwork,
        device: str = "cuda",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        batch_size: int = 256,
    ):
        self.env = env
        self.network = network.to(device)
        self.device = device

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Optimizer
        self.optimizer = optim.Adam(network.parameters(), lr=learning_rate)

        # Statistics
        self.total_steps = 0
        self.episode_rewards = []

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation

        Uses CUDA kernels if available, otherwise PyTorch
        """
        if CUDA_KERNELS_AVAILABLE and rewards.is_cuda:
            return market_sim_cuda.compute_gae(
                rewards, values, dones, self.gamma, self.gae_lambda
            )
        else:
            # PyTorch implementation
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)
            gae = 0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae

                advantages[t] = gae
                returns[t] = gae + values[t]

            return advantages, returns

    def collect_rollout(self, num_steps: int):
        """Collect rollout data from environment"""
        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        obs, _ = self.env.reset()
        obs = torch.FloatTensor(obs).to(self.device)

        for _ in range(num_steps):
            with torch.no_grad():
                policy_logits, value, _ = self.network(obs)

                # Sample action
                action_dist = torch.distributions.Categorical(logits=policy_logits)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(action.cpu().numpy())

            # Store transition
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(torch.FloatTensor([reward]).to(self.device))
            values.append(value)
            dones.append(torch.FloatTensor([float(done or truncated)]).to(self.device))

            obs = torch.FloatTensor(next_obs).to(self.device)
            self.total_steps += 1

            if done or truncated:
                obs, _ = self.env.reset()
                obs = torch.FloatTensor(obs).to(self.device)

        # Stack all transitions
        return {
            "observations": torch.stack(observations),
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "rewards": torch.stack(rewards),
            "values": torch.stack(values),
            "dones": torch.stack(dones),
        }

    def update(self, rollout_data: Dict):
        """Perform PPO update"""
        obs = rollout_data["observations"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["log_probs"]
        rewards = rollout_data["rewards"]
        old_values = rollout_data["values"]
        dones = rollout_data["dones"]

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of updates
        for _ in range(self.num_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(obs))

            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward pass
                policy_logits, values, _ = self.network(batch_obs)

                # Policy loss
                action_dist = torch.distributions.Categorical(logits=policy_logits)
                log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def train(self, total_timesteps: int, rollout_steps: int = 2048, log_interval: int = 10):
        """Main training loop"""
        num_updates = total_timesteps // rollout_steps

        print(f"Starting training for {total_timesteps} timesteps ({num_updates} updates)")
        print(f"Device: {self.device}")
        print(f"CUDA kernels: {'Available' if CUDA_KERNELS_AVAILABLE else 'Not available'}")

        for update in range(num_updates):
            start_time = time.time()

            # Collect rollout
            rollout_data = self.collect_rollout(rollout_steps)

            # Update policy
            update_info = self.update(rollout_data)

            elapsed = time.time() - start_time
            fps = rollout_steps / elapsed

            if (update + 1) % log_interval == 0:
                mean_reward = rollout_data["rewards"].mean().item()
                print(f"Update {update + 1}/{num_updates} | "
                      f"Steps: {self.total_steps} | "
                      f"FPS: {fps:.0f} | "
                      f"Mean Reward: {mean_reward:.4f} | "
                      f"Policy Loss: {update_info['policy_loss']:.4f} | "
                      f"Value Loss: {update_info['value_loss']:.4f}")


def main():
    """Example training script"""
    # Configuration
    from market_sim_c.python.market_sim import make_env, MarketSimConfig

    config = MarketSimConfig(
        num_assets=10,
        num_agents=1,
        max_steps=1000,
    )

    # Create environment (use Python version for now until C version is compiled)
    env = make_env(config, use_python=True)

    # Create network
    obs_size = 5 * 256 + config.num_assets + 3
    action_size = 1 + 2 * config.num_assets

    network = MarketPPONetwork(obs_size, action_size)

    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = PPOTrainer(env, network, device=device)

    # Train
    trainer.train(total_timesteps=1_000_000, rollout_steps=2048)


if __name__ == "__main__":
    main()
