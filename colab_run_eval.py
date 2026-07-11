import argparse
import inspect
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--decision-log", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    os.chdir(str(repo_root))

    import eval as eval_mod

    print("Loaded eval module from:", eval_mod.__file__)
    print("Available functions:", [name for name in dir(eval_mod) if name.startswith("run")])

    if hasattr(eval_mod, "run_from_colab"):
        eval_mod.run_from_colab(
            model_path=args.model_path,
            episodes=args.episodes,
            config_path=args.config,
            output_csv=args.output_csv,
            decision_log_path=args.decision_log,
        )
    elif hasattr(eval_mod, "run_eval"):
        eval_mod.run_eval(
            model_path=args.model_path,
            episodes=args.episodes,
            config_path=args.config,
            output_csv=args.output_csv,
            decision_log_path=args.decision_log,
        )
    else:
        raise AttributeError("Neither run_from_colab nor run_eval is available in eval.py")


if __name__ == "__main__":
    main()
