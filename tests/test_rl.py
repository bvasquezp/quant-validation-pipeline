import numpy as np
import torch

from src.config import load_config
from src.models.rl import CausalTCNActorCritic, FxTradingEnv, evaluate_policy, train_ppo


def test_rl_environment_applies_costs_and_actions():
    features = np.zeros((20, 3), dtype=np.float32)
    returns = np.full(20, 0.001, dtype=np.float32)
    env = FxTradingEnv(features, returns, lookback=4, spread_pips=1.0, pip_size=0.0001)

    observation = env.reset()
    next_observation, reward, done, info = env.step(2)

    assert observation.shape == (4, 5)
    assert next_observation.shape == observation.shape
    assert done is False
    assert info["cost"] > 0
    assert reward < info["gross_return"]


def test_ppo_trains_and_emits_only_discrete_actions():
    config = load_config("config/pipeline_config.yaml").models.rl
    config.lookback = 8
    config.n_epochs = 1
    config.update_epochs = 1
    config.batch_size = 16
    features = np.random.default_rng(4).normal(size=(40, 3)).astype(np.float32)
    returns = np.random.default_rng(5).normal(0, 0.001, size=40).astype(np.float32)
    env = FxTradingEnv(features, returns, lookback=config.lookback)

    policy = train_ppo(env, config, seed=3)
    result = evaluate_policy(policy, FxTradingEnv(features, returns, lookback=config.lookback))

    assert len(result["returns"]) == len(result["actions"])
    assert set(result["actions"].tolist()) <= {0, 1, 2}


def test_rl_tcn_preserves_observation_length_causally():
    model = CausalTCNActorCritic(n_features=5, hidden=8)
    logits, value = model(torch.zeros((2, 16, 5)))

    assert logits.shape == (2, 3)
    assert value.shape == (2,)
