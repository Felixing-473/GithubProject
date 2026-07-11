import argparse
import csv
import numpy as np
import torch
from env.flood_env import FloodEvacuationEnv
from agents.ppo import ActorCritic


def load_policy(path, env):
    model = ActorCritic(env.observation_space.shape[0], [3, 3, 3])
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def act(model, obs):
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0))
    actions = []
    for logit in logits:
        probs = torch.softmax(logit, dim=-1)
        a = int(torch.argmax(probs).item())
        actions.append(a)
    return actions


def run_episode(env, model=None, seed=42, verbose=True, decision_log_path=None):
    obs = env.reset(pre_disaster_action=0)
    done = False
    total_reward = 0.0
    step_count = 0
    logs = []

    # log pre-disaster setup once
    pre_action = getattr(env, "last_pre_disaster_action", None)
    if verbose:
        print(f"[PRE] pre_disaster_action={pre_action}")

    while not done:
        # snapshot state before policy action
        before = {
            "step": step_count + 1,
            "pre_disaster_action": pre_action,
            "flood_target": env.flood_target,
            "hub_states": {
                k: {
                    "severe": h.current_severe,
                    "medium": h.current_medium,
                    "normal": h.current_normal,
                    "food": round(h.food, 2),
                    "med": round(h.med, 2),
                    "available": h.is_available,
                    "stage": h.stage_state,
                }
                for k, h in env.hubs.items()
            },
            "city_states": {
                k: {
                    "severe": c.current_severe,
                    "medium": c.current_medium,
                    "normal": c.current_normal,
                    "severe_remaining": c.severe_remaining,
                    "medium_remaining": c.medium_remaining,
                    "normal_remaining": c.normal_remaining,
                }
                for k, c in env.cities.items()
            },
        }

        if model is None:
            action = env.action_space["step_action"].sample()
        else:
            action = act(model, obs)

        obs, reward, done, info = env.step({"step_action": action, "pre_action": 0})
        total_reward += reward
        step_count += 1

        after = {
            "step": step_count,
            "reward": float(reward),
            "done": bool(done),
            "action": list(action),
            "flood_target": env.flood_target,
            "hub_states": {
                k: {
                    "severe": h.current_severe,
                    "medium": h.current_medium,
                    "normal": h.current_normal,
                    "food": round(h.food, 2),
                    "med": round(h.med, 2),
                    "available": h.is_available,
                    "stage": h.stage_state,
                }
                for k, h in env.hubs.items()
            },
            "city_states": {
                k: {
                    "severe": c.current_severe,
                    "medium": c.current_medium,
                    "normal": c.current_normal,
                    "severe_remaining": c.severe_remaining,
                    "medium_remaining": c.medium_remaining,
                    "normal_remaining": c.normal_remaining,
                }
                for k, c in env.cities.items()
            },
        }

        log_row = {**before, **after}
        logs.append(log_row)

        if verbose:
            print(
                f"[STEP {step_count}] action={action} reward={reward:.2f} flood_target={env.flood_target} "
                f"hub_B1={before['hub_states']['B1']} hub_B2={before['hub_states']['B2']} hub_B3={before['hub_states']['B3']}"
            )

    if decision_log_path:
        with open(decision_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "pre_disaster_action",
                    "flood_target",
                    "action",
                    "reward",
                    "done",
                    "hub_states",
                    "city_states",
                ],
            )
            writer.writeheader()
            for row in logs:
                writer.writerow({
                    "step": row["step"],
                    "pre_disaster_action": row["pre_disaster_action"],
                    "flood_target": row["flood_target"],
                    "action": row["action"],
                    "reward": row["reward"],
                    "done": row["done"],
                    "hub_states": row["hub_states"],
                    "city_states": row["city_states"],
                })

    summary = {
        "steps": step_count,
        "total_reward": float(total_reward),
        "hub_waiting": {k: h.total_people for k, h in env.hubs.items()},
        "city_population": {k: c.current_normal + c.current_medium + c.current_severe for k, c in env.cities.items()},
        "flood_target": env.flood_target,
        "decision_log": logs,
    }
    return summary


def run_eval(model_path=None, episodes=1, config_path=None, config=None, output_csv=None, decision_log_path=None):
    config_file = config_path or config or "configs/default.json"
    env = FloodEvacuationEnv(config_path=config_file, seed=42)
    model = load_policy(model_path, env) if model_path else None
    results = []
    for ep in range(episodes):
        summary = run_episode(env, model=model, verbose=True, decision_log_path=decision_log_path)
        results.append(summary)

    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["steps", "total_reward", "hub_waiting", "city_population", "flood_target"])
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    print("Simulation results:")
    for idx, row in enumerate(results, 1):
        print(f"Episode {idx}: steps={row['steps']}, total_reward={row['total_reward']:.2f}")
        print(f"  hub_waiting={row['hub_waiting']}")
        print(f"  city_population={row['city_population']}")
        print(f"  flood_target={row['flood_target']}")

    if len(results) > 1:
        rewards = [r["total_reward"] for r in results]
        print(f"Average reward over {len(results)} episodes: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--decision-log", default=None)
    args = parser.parse_args()
    run_eval(model_path=args.model_path, episodes=args.episodes, config_path=args.config, output_csv=args.output_csv, decision_log_path=args.decision_log)


def run_from_colab(model_path, episodes=1, config_path="configs/default.json", output_csv=None, decision_log_path=None):
    run_eval(model_path=model_path, episodes=episodes, config_path=config_path, output_csv=output_csv, decision_log_path=decision_log_path)
