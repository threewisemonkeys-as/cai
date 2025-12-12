"""CLI entry point for running the Froggy bug generation pipeline.

Usage:
    python -m froggy_buggen [--config CONFIG_PATH]
"""

import fire
from dotenv import load_dotenv

from froggy_buggen.pipeline import regular

load_dotenv()

if __name__ == "__main__":
    fire.Fire(regular)
