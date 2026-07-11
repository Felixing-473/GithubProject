import csv
import itertools
from env.flood_env import FloodEvacuationEnv


def run_config(config_path, pre_action=0, steps=72):
    env = FloodEvacuationEnv(config_path=config_path, seed=0)
    env.reset(pre_disaster_action=pre_action)
    total_reward = 0.0
    for _ in range(steps):
        action = {"step_action": env.action_space["step_action"].sample(), "pre_action": pre_action}
        obs, r, done, info = env.step(action)
        total_reward += r
        if done:
            break
    return total_reward


def main():
    configs = ["configs/default.json", "configs/human_low.json", "configs/human_high.json"]
    results = []
    for cfg in configs:
        r = run_config(cfg)
        results.append((cfg, r))
    # write CSV
    with open("param_search_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "total_reward"])
        for row in results:
            w.writerow(row)
    print("Param search done. Results saved to param_search_results.csv")


if __name__ == "__main__":
    main()
