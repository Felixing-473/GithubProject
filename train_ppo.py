import os
import argparse
from agents.ppo import PPOTrainer
from env.flood_env import FloodEvacuationEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", default="checkpoints/ppo.pt")
    args = parser.parse_args()

    env = FloodEvacuationEnv(config_path=args.config, seed=args.seed)
    trainer = PPOTrainer(
        env,
        lr=args.lr,
        batch_size=args.batch_size,
        minibatch=args.minibatch,
    )
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    trainer.train(total_steps=args.total_steps, save_path=args.save_path)


if __name__ == "__main__":
    main()
