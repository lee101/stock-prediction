"""Training entrypoint for the third-generation PufferLib trading pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import gymnasium as gym
import torch

try:  # Provide gym alias for libraries that still import gym directly.
    import gym  # type: ignore[unused-import]
except ModuleNotFoundError:  # pragma: no cover - depends on local install
    import sys

    sys.modules["gym"] = gym  # gymnasium alias for old API consumers

from .envs.market_env import MarketEnv, MarketEnvConfig
from .models import MarketPolicy, PolicyConfig


@dataclass
class PPOConfig:
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.8
    max_grad_norm: float = 1.0
    batch_size: int = 262_144
    minibatch_size: int = 65_536
    update_epochs: int = 3
    bptt_horizon: int = 512
    use_rnn: bool = False
    optimizer: str = "adam"
    compile: bool = False
    compile_mode: str = "max-autotune"
    torch_deterministic: bool = False
    precision: str = "float32"
    cpu_offload: bool = False


@dataclass
class VecConfig:
    backend: str = "Serial"
    num_envs: int = 16
    num_workers: int = 1
    seed: int = 1337


class MetricsCollector(gym.Wrapper):
    """Wrap an environment to accumulate step-level cost metrics."""

    def __init__(self, env: gym.Env, tracked: Iterable[str]):
        super().__init__(env)
        self.tracked = tuple(tracked)
        self._metrics: Dict[str, float] = {key: 0.0 for key in self.tracked}
        self._steps = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        for key in self.tracked:
            if key in info:
                self._metrics[key] += float(info[key])
        self._steps += 1
        return obs, reward, terminated, truncated, info

    def snapshot_and_reset(self) -> Dict[str, float]:
        data = {key: value for key, value in self._metrics.items()}
        data["steps"] = float(self._steps)
        self._metrics = {key: 0.0 for key in self.tracked}
        self._steps = 0
        return data


def _import_pufferlib_module(name: str):
    import importlib
    import sys

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as first_error:  # pragma: no cover - depends on local install
        repo_root = Path(__file__).resolve().parents[1] / "PufferLib"
        repo_path = str(repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as second_error:
            raise ModuleNotFoundError(
                f"Unable to import '{name}'. Ensure PufferLib is installed or available at {repo_root}."
            ) from second_error


def _str_to_bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, received '{value}'")


def _device_string(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return device.type


def _resolve_device(device: str) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return requested


def _build_env_creator(cfg: MarketEnvConfig, collectors: List[MetricsCollector]):
    tracked = ("trading_cost", "financing_cost", "deleverage_notional")
    emulation = _import_pufferlib_module("pufferlib.emulation")

    def _gym_env() -> gym.Env:
        env_cfg = replace(cfg)
        env = MarketEnv(cfg=env_cfg)
        wrapper = MetricsCollector(env, tracked)
        collectors.append(wrapper)
        return wrapper

    def _puffer_env(*, buf=None, seed: Optional[int] = None, **kwargs) -> gym.Env:
        # PufferLib vector backends pass shared-memory buffers and seeds; forward them explicitly.
        return emulation.GymnasiumPufferEnv(env_creator=_gym_env, buf=buf, seed=seed)

    return _puffer_env


def _build_vecenv(vec_cfg: VecConfig, env_creator, device: torch.device):
    vector = _import_pufferlib_module("pufferlib.vector")
    env_creators = [env_creator] * vec_cfg.num_envs
    env_args = [[] for _ in range(vec_cfg.num_envs)]
    env_kwargs = [{} for _ in range(vec_cfg.num_envs)]
    backend = getattr(vector, vec_cfg.backend, vec_cfg.backend)
    vecenv = vector.make(
        env_creators,
        env_args=env_args,
        env_kwargs=env_kwargs,
        backend=backend,
        num_envs=vec_cfg.num_envs,
        seed=vec_cfg.seed,
        num_workers=vec_cfg.num_workers,
    )
    vecenv.device = device
    return vecenv


def _update_train_config(train_cfg: MutableMapping[str, Any], ppo_cfg: PPOConfig, *, device: torch.device, seed: int) -> None:
    train_cfg.update(
        total_timesteps=ppo_cfg.total_timesteps,
        learning_rate=ppo_cfg.learning_rate,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_coef=ppo_cfg.clip_coef,
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        batch_size=ppo_cfg.batch_size,
        minibatch_size=ppo_cfg.minibatch_size,
        update_epochs=ppo_cfg.update_epochs,
        bptt_horizon=ppo_cfg.bptt_horizon,
        use_rnn=ppo_cfg.use_rnn,
        optimizer=ppo_cfg.optimizer,
        compile=ppo_cfg.compile,
        compile_mode=ppo_cfg.compile_mode,
        torch_deterministic=ppo_cfg.torch_deterministic,
        precision=ppo_cfg.precision,
        cpu_offload=ppo_cfg.cpu_offload,
        device=_device_string(device),
        seed=seed,
    )


def _flush_metrics(collectors: Iterable[MetricsCollector]) -> Dict[str, float]:
    aggregate: Dict[str, float] = {}
    for collector in collectors:
        snapshot = collector.snapshot_and_reset()
        for key, value in snapshot.items():
            aggregate[key] = aggregate.get(key, 0.0) + float(value)
    return aggregate


def _log_epoch_metrics(epoch: int, metrics: Mapping[str, float]) -> None:
    if not metrics:
        return
    steps = metrics.get("steps", 0.0) or 1.0
    trading = metrics.get("trading_cost", 0.0)
    financing = metrics.get("financing_cost", 0.0)
    deleverage = metrics.get("deleverage_notional", 0.0)
    logging.info(
        "epoch=%d metrics: trading_cost=%.6f financing_cost=%.6f deleverage_notional=%.6f steps=%.0f",
        epoch,
        trading,
        financing,
        deleverage,
        steps,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MarketEnv with PuffeRL")
    parser.add_argument("--data-root", type=str, default="trainingdata", help="Directory containing OHLC CSV files")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to prioritise when loading CSV data")
    parser.add_argument("--mode", type=str, default="open_close", choices=["open_close", "event", "maxdiff"], help="Market execution mode")
    parser.add_argument("--is-crypto", type=_str_to_bool, default=False, help="Whether the asset is crypto (disables leverage financing)")
    parser.add_argument("--context-len", type=int, default=128, help="Number of timesteps in each observation window")
    parser.add_argument("--horizon", type=int, default=1, help="Reward horizon in steps")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Scalar multiplier applied to rewards")
    parser.add_argument("--intraday-leverage", type=float, default=4.0, help="Maximum intraday leverage multiple")
    parser.add_argument("--overnight-leverage", type=float, default=2.0, help="Maximum overnight leverage multiple")
    parser.add_argument("--annual-leverage-rate", type=float, default=0.0675, help="Annual financing rate applied to positions above 1x")
    parser.add_argument("--trading-fee", type=float, default=0.0005, help="Per-notional trading fee for equities")
    parser.add_argument("--crypto-fee", type=float, default=0.0015, help="Per-notional trading fee for crypto assets")
    parser.add_argument("--slip-bps", type=float, default=1.5, help="Linear slippage in basis points per |Δposition|")
    parser.add_argument("--maxdiff-limit-scale", type=float, default=0.05, help="Scale for limit offsets in MaxDiff mode (fraction of open price)")
    parser.add_argument("--maxdiff-deadband", type=float, default=0.05, help="Minimum |direction| required before placing MaxDiff bracket orders")
    parser.add_argument("--normalize-returns", type=_str_to_bool, default=True, help="Normalize returns in observations")
    parser.add_argument("--inv-penalty", type=float, default=0.0, help="Inventory L2 penalty coefficient")
    parser.add_argument("--seed", type=int, default=1337, help="PRNG seed")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (e.g. 'cuda', 'cuda:1', 'cpu')")
    parser.add_argument("--backend", type=str, default="Serial", choices=["Serial", "Multiprocessing"], help="Vector environment backend")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environment replicas")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker processes for vector backend")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000, help="Total agent timesteps")
    parser.add_argument("--batch-size", type=int, default=262_144, help="Global batch size per PPO epoch")
    parser.add_argument("--minibatch-size", type=int, default=65_536, help="Minibatch size per PPO update")
    parser.add_argument("--update-epochs", type=int, default=3, help="Number of PPO epochs per batch")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.005, help="Entropy regularisation coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.8, help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--bptt-horizon", type=int, default=512, help="Truncated BPTT horizon")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer (adam or muon)")
    parser.add_argument("--compile", type=_str_to_bool, default=False, help="Use torch.compile for policy")
    parser.add_argument("--compile-mode", type=str, default="max-autotune", help="torch.compile mode")
    parser.add_argument("--torch-deterministic", type=_str_to_bool, default=False, help="Enable deterministic CuDNN kernels")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "bfloat16"], help="AMP precision")
    parser.add_argument("--cpu-offload", type=_str_to_bool, default=False, help="Enable CPU offload for experience buffers")
    parser.add_argument("--log-json", type=str, default=None, help="Optional path to dump final summary JSON")
    parser.add_argument("--log-level", type=str, default="INFO", help="Python logging level")
    return parser.parse_args(argv)


def build_configs(args: argparse.Namespace) -> tuple[MarketEnvConfig, PPOConfig, VecConfig, torch.device]:
    device = _resolve_device(args.device)
    env_cfg = MarketEnvConfig(
        context_len=args.context_len,
        horizon=args.horizon,
        mode=args.mode,
        data_root=args.data_root,
        symbol=args.symbol,
        normalize_returns=args.normalize_returns,
        trading_fee=args.trading_fee,
        crypto_trading_fee=args.crypto_fee,
        slip_bps=args.slip_bps,
        annual_leverage_rate=args.annual_leverage_rate,
        intraday_leverage_max=args.intraday_leverage,
        overnight_leverage_max=args.overnight_leverage,
        inv_penalty=args.inv_penalty,
        action_space="continuous",
        reward_scale=args.reward_scale,
        is_crypto=args.is_crypto,
        seed=args.seed,
        device=device.type,
        maxdiff_limit_scale=args.maxdiff_limit_scale,
        maxdiff_deadband=args.maxdiff_deadband,
    )
    ppo_cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        bptt_horizon=args.bptt_horizon,
        use_rnn=False,
        optimizer=args.optimizer,
        compile=args.compile,
        compile_mode=args.compile_mode,
        torch_deterministic=args.torch_deterministic,
        precision=args.precision,
        cpu_offload=args.cpu_offload,
    )
    vec_cfg = VecConfig(
        backend=args.backend,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    return env_cfg, ppo_cfg, vec_cfg, device


def train(args: argparse.Namespace) -> Dict[str, Any]:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    env_cfg, ppo_cfg, vec_cfg, device = build_configs(args)

    collectors: List[MetricsCollector] = []
    env_creator = _build_env_creator(env_cfg, collectors)
    vecenv = _build_vecenv(vec_cfg, env_creator, device)

    pufferl = _import_pufferlib_module("pufferlib.pufferl")
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        base_cfg = pufferl.load_config("default")
    finally:
        sys.argv = original_argv
    train_cfg = dict(base_cfg["train"])
    _update_train_config(train_cfg, ppo_cfg, device=device, seed=args.seed)
    train_cfg["env"] = "pufferlibtraining3.market_env"

    model_cfg = PolicyConfig()
    policy = MarketPolicy(vecenv.driver_env, model_cfg).to(device)

    trainer = pufferl.PuffeRL(train_cfg, vecenv, policy)

    epoch = 0
    final_logs: Dict[str, float] = {}
    try:
        while trainer.global_step < train_cfg["total_timesteps"]:
            trainer.evaluate()
            logs = trainer.train()
            if logs:
                merged = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
                final_logs.update(merged)
                logging.info("epoch=%d global_step=%d %s", epoch, trainer.global_step, json.dumps(merged))
            metrics = _flush_metrics(collectors)
            _log_epoch_metrics(epoch, metrics)
            epoch += 1

        trainer.print_dashboard()
        summary_logs = trainer.mean_and_log()
        if summary_logs:
            merged = {k: float(v) for k, v in summary_logs.items() if isinstance(v, (int, float))}
            final_logs.update(merged)
    finally:
        model_path = trainer.close()
        try:
            vecenv.close()
        except Exception:  # pragma: no cover - defensive cleanup
            pass

    summary: Dict[str, Any] = {
        "run_id": getattr(trainer.logger, "run_id", "local"),
        "model_path": model_path,
        "final_logs": final_logs,
        "train_config": train_cfg,
        "env_config": dataclasses.asdict(env_cfg),
    }

    log_path = args.log_json
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        logging.info("Wrote summary JSON to %s", path)

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
