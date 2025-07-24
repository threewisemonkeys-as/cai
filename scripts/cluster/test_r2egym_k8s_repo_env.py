#!/usr/bin/env python3
"""
Self-contained script to instantiate a RepoEnv with Kubernetes backend and call reset on it.
"""

import json
from pathlib import Path

from r2egym.agenthub.environment.env import RepoEnv, EnvArgs

CUR_DIR = Path(__file__).parent

def main():
    ds = json.load(open(CUR_DIR / "r2egym_sample.json", "r"))
    
    print("Creating RepoEnv with Kubernetes backend...")
    
    env = RepoEnv(
        args=EnvArgs(ds),
        backend="kubernetes",
        verbose=True,
        step_timeout=90,
        reward_timeout=300
    )
    
    print("RepoEnv created successfully!")
    print(f"Backend: {env.backend}")
    print(f"Docker image: {env.runtime.docker_image}")
    
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