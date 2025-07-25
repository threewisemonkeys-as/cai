#!/usr/bin/env python3
"""
Self-contained script to instantiate a RepoEnv with Kubernetes backend and call reset on it.
"""

import json
from pathlib import Path

from rllm.environments.swe.swe import SWEEnv

CUR_DIR = Path(__file__).parent

def main():
    entry = json.load(open(CUR_DIR / "r2egym_sample.json", "r"))

    print("Creating SWEEnv with Kubernetes backend...")

    env = SWEEnv(
        entry=entry,
        backend="kubernetes",
        verbose=True,
        step_timeout=90,
        reward_timeout=300
    )

    print("SWEEnv created successfully!")
    print(f"Backend: {env.backend}")
    
    # Call reset on the environment
    print("\nCalling reset on the environment...")
    observation = env.reset()
    
    print(f"Reset completed. Observation: {observation}")
    
    # Clean up
    print("\nClosing environment...")
    env.close()
    print("Environment closed successfully!")

if __name__ == "__main__":
    main()