"""Entorno de trading y PPO discreto para la política EUR/USD."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.features.fractional_diff import FoldFeatureSet
from src.ledger.trial_ledger import TrialRecord, log_trial


ACTION_TO_POSITION = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


class FxTradingEnv:
    """Simulador causal mínimo: acción en t, retorno neto en t."""

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        lookback: int = 64,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.2,
        pip_size: float = 0.0001,
        turnover_penalty: float = 0.0,
        drawdown_penalty: float = 0.0,
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32)
        if features.ndim != 2 or len(features) != len(returns):
            raise ValueError("features and returns must have equal 2-D/1-D length")
        if len(returns) <= lookback:
            raise ValueError("RL episode needs more observations than lookback")
        self.features = features
        self.returns = returns
        self.lookback = lookback
        self.cost_per_turn = (spread_pips + slippage_pips) * pip_size
        self.turnover_penalty = turnover_penalty
        self.drawdown_penalty = drawdown_penalty
        self.n_features = features.shape[1] + 2
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = self.lookback
        self.position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        return self._observation()

    def _observation(self) -> np.ndarray:
        window = self.features[self.t - self.lookback : self.t]
        drawdown = self.equity / max(self.peak_equity, 1e-12) - 1.0
        state = np.tile(
            np.array([self.position, drawdown], dtype=np.float32),
            (self.lookback, 1),
        )
        return np.concatenate([window, state], axis=1)

    def step(self, action: int):
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action: {action}")
        new_position = float(ACTION_TO_POSITION[action])
        turnover = abs(new_position - self.position)
        cost = turnover * self.cost_per_turn
        net_return = new_position * float(self.returns[self.t]) - cost
        self.equity *= float(np.exp(net_return))
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = max(0.0, 1.0 - self.equity / self.peak_equity)
        reward = (
            net_return
            - self.turnover_penalty * turnover
            - self.drawdown_penalty * drawdown
        )
        self.position = new_position
        self.t += 1
        done = self.t >= len(self.returns)
        info = {
            "position": self.position,
            "gross_return": new_position * float(self.returns[self.t - 1]),
            "cost": cost,
            "net_return": net_return,
            "equity": self.equity,
        }
        return self._observation() if not done else np.zeros_like(self._observation()), reward, done, info


class CausalTCNActorCritic(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0.0),
            nn.Conv1d(n_features, hidden, kernel_size=3, padding=0, dilation=1),
            nn.Tanh(),
            nn.ConstantPad1d((4, 0), 0.0),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=0, dilation=2),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, 3)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, observation: torch.Tensor):
        encoded = self.encoder(observation.transpose(1, 2))
        latent = encoded[:, :, -1]
        return self.actor(latent), self.critic(latent).squeeze(-1)


def _gae(rewards, values, dones, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    next_value = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        delta = rewards[i] + gamma * next_value * (1.0 - dones[i]) - values[i]
        running = delta + gamma * gae_lambda * (1.0 - dones[i]) * running
        advantages[i] = running
        next_value = values[i]
    returns = advantages + values
    return advantages, returns


def train_ppo(
    env: FxTradingEnv,
    config: Any,
    seed: int = 42,
) -> CausalTCNActorCritic:
    """Entrena PPO on-policy sobre un episodio histórico."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = CausalTCNActorCritic(env.n_features)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)

    for _ in range(config.n_epochs):
        observations, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        observation = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(observation[None], dtype=torch.float32)
            logits, value = policy(obs_tensor)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            next_observation, reward, done, _ = env.step(int(action.item()))
            observations.append(observation)
            actions.append(int(action.item()))
            rewards.append(reward)
            values.append(float(value.item()))
            log_probs.append(float(distribution.log_prob(action).item()))
            dones.append(float(done))
            observation = next_observation

        advantages, target_values = _gae(
            np.asarray(rewards, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            config.gamma,
            config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        obs_tensor = torch.tensor(np.asarray(observations), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.long)
        old_log_tensor = torch.tensor(log_probs, dtype=torch.float32)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        target_tensor = torch.tensor(target_values, dtype=torch.float32)

        for _ in range(config.update_epochs):
            logits, value = policy(obs_tensor)
            distribution = Categorical(logits=logits)
            new_log = distribution.log_prob(action_tensor)
            ratio = torch.exp(new_log - old_log_tensor)
            clipped = torch.clamp(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
            actor_loss = -torch.min(ratio * adv_tensor, clipped * adv_tensor).mean()
            critic_loss = (target_tensor - value).pow(2).mean()
            loss = (
                actor_loss
                + config.value_coef * critic_loss
                - config.entropy_coef * distribution.entropy().mean()
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()
    return policy


def evaluate_policy(policy, env: FxTradingEnv) -> Dict[str, Any]:
    observation = env.reset()
    done = False
    returns = []
    actions = []
    while not done:
        with torch.no_grad():
            logits, _ = policy(torch.tensor(observation[None], dtype=torch.float32))
        action = int(torch.argmax(logits, dim=-1).item())
        observation, _, done, info = env.step(action)
        returns.append(info["net_return"])
        actions.append(action)
    return {"returns": np.asarray(returns), "actions": np.asarray(actions)}


def ppo_trading(
    fold_feature_sets: list[FoldFeatureSet],
    rl_config: Any,
    seed: int = 42,
    ledger_path: Optional[str] = None,
    ledger_metadata: Optional[Dict[str, Any]] = None,
    annualization: float = 1.0,
) -> Dict[str, Any]:
    """Entrena/evalúa una política independiente por fold."""
    all_returns = []
    all_actions = []
    all_timestamps = []
    for item in fold_feature_sets:
        if len(item.train_features) <= rl_config.lookback or item.test_features.empty:
            continue
        train_env = FxTradingEnv(
            item.train_features.values,
            item.train_target.values,
            lookback=rl_config.lookback,
            spread_pips=rl_config.spread_pips,
            slippage_pips=rl_config.slippage_pips,
            pip_size=rl_config.pip_size,
            turnover_penalty=rl_config.turnover_penalty,
            drawdown_penalty=rl_config.drawdown_penalty,
        )
        policy = train_ppo(train_env, rl_config, seed + item.fold_id)
        test_env = FxTradingEnv(
            item.test_features.values,
            item.test_target.values,
            lookback=rl_config.lookback,
            spread_pips=rl_config.spread_pips,
            slippage_pips=rl_config.slippage_pips,
            pip_size=rl_config.pip_size,
            turnover_penalty=rl_config.turnover_penalty,
            drawdown_penalty=rl_config.drawdown_penalty,
        )
        evaluation = evaluate_policy(policy, test_env)
        all_returns.append(evaluation["returns"])
        all_actions.append(evaluation["actions"])
        all_timestamps.extend(item.test_features.index.to_numpy()[rl_config.lookback:])

    returns_oos = np.concatenate(all_returns) if all_returns else np.array([])
    actions_oos = np.concatenate(all_actions) if all_actions else np.array([], dtype=int)
    sharpe = float(
        returns_oos.mean() / returns_oos.std(ddof=1) * np.sqrt(annualization)
        if len(returns_oos) > 1 and returns_oos.std(ddof=1) > 0 else 0.0
    )
    hyperparams = {"architecture": "ppo_causal_tcn", **vars(rl_config)}
    if ledger_metadata:
        hyperparams["feature_metadata"] = ledger_metadata
    result = {
        "model_name": "ppo_trading",
        "returns_oos": returns_oos,
        "oos_timestamps": np.asarray(all_timestamps),
        "actions_oos": actions_oos,
        "sharpe_oos": sharpe,
        "sharpe_is": 0.0,
        "n_observations": len(returns_oos),
        "hyperparams": hyperparams,
    }
    if ledger_path:
        record = TrialRecord(
            agent="rl",
            model_name="ppo_trading",
            config_hash=TrialRecord.hash_config(hyperparams),
            hyperparams=json.dumps(hyperparams, default=str),
            sharpe_is=0.0,
            sharpe_oos=sharpe,
            n_observations=len(returns_oos),
            d_used=None,
            d_selection_method=ledger_metadata.get("d_selection_method", "") if ledger_metadata else "",
            annualization_factor=annualization,
            frequency=ledger_metadata.get("frequency", "") if ledger_metadata else "",
            config_version=ledger_metadata.get("config_version", "") if ledger_metadata else "",
        )
        log_trial(record, ledger_path)
        result["trial_id"] = record.trial_id
    return result
