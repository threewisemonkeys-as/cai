"""CLI entry point for running the Debug-Gym bug generation pipeline."""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from debug_gym_bundle.pipeline import regular

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

__all__ = ["regular"]


if __name__ == "__main__":
    import fire

    fire.Fire(regular)
