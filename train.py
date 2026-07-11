"""
Simple runner to demonstrate the environment. Runs a random policy for one episode.
"""
from env.flood_env import FloodEvacuationEnv
import random


def run_random_episode(steps=72):
    env = FloodEvacuationEnv(config_path="configs/default.json", seed=42)
    obs = env.reset()
    done = False
    total_reward = 0.0
    for _ in range(steps):
        action = {
            "step_action": env.action_space["step_action"].sample(),
            "pre_action": 0,
        }
        obs, r, done, info = env.step(action)
        total_reward += r
        if done:
            break
    print("Episode done. Total reward:", total_reward)


if __name__ == "__main__":
    run_random_episode()
