import argparse
import subprocess
import sys

from pipeline import train_and_save_models


def main():
    parser = argparse.ArgumentParser(description="Train forecasting models and optionally start the API.")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI after training.")
    parser.add_argument("--port", type=int, default=8000, help="API port.")
    parser.add_argument("--limit-states", type=int, default=None, help="Train only the first N states for a smoke test.")
    args = parser.parse_args()

    train_and_save_models(limit_states=args.limit_states)

    if args.serve:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(args.port),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()

