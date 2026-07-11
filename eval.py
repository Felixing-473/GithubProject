import torch
import numpy as np
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


def run_eval(model_path=None, episodes=10):
    env = FloodEvacuationEnv(config_path="configs/default.json", seed=42)
    if model_path:
        model = load_policy(model_path, env)
    else:
        model = None
    results = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if model is None:
                action = env.action_space["step_action"].sample()
            else:
                action = act(model, obs)
            obs, r, done, info = env.step({"step_action": action, "pre_action": 0})
            total_reward += r
        results.append(total_reward)
    print("Eval results avg:", np.mean(results), "std:", np.std(results))


if __name__ == "__main__":
    run_eval()
