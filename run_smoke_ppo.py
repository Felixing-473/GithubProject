from agents.ppo import PPOTrainer
from env.flood_env import FloodEvacuationEnv
import os


def main():
    env = FloodEvacuationEnv(config_path="configs/default.json", seed=0)
    trainer = PPOTrainer(env, batch_size=512, minibatch=64)
    os.makedirs("checkpoints", exist_ok=True)
    print("Starting smoke PPO training (2000 steps)...")
    trainer.train(total_steps=2000, save_path="checkpoints/ppo_smoke.pt")
    print("Training finished. Checkpoint saved to checkpoints/ppo_smoke.pt")


if __name__ == "__main__":
    main()
