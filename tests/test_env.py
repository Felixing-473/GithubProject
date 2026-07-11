import pytest
from env.flood_env import FloodEvacuationEnv


def test_env_runs_smoke():
    env = FloodEvacuationEnv(config_path="configs/default.json", seed=123)
    obs = env.reset(pre_disaster_action=0)
    assert obs.shape[0] == 29
    done = False
    for _ in range(10):
        action = {"step_action": env.action_space["step_action"].sample(), "pre_action": 0}
        obs, r, done, info = env.step(action)
        assert isinstance(r, float)
        if done:
            break


def test_reward_negative_when_severe_waiting():
    env = FloodEvacuationEnv(config_path="configs/default.json", seed=1)
    env.reset(pre_disaster_action=0)
    # directly compute reward with initial severe patients present
    r = env._compute_reward()
    assert isinstance(r, float)
    assert r < 0
