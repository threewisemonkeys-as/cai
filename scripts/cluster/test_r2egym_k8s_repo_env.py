#!/usr/bin/env python3
"""
Self-contained script to instantiate a RepoEnv with Kubernetes backend and call reset on it.
"""

from datasets import load_dataset
from r2egym.agenthub.environment.env import RepoEnv, EnvArgs

def main():
    ds = load_dataset("R2E-Gym/R2E-Gym", split="train")
    
    print("Creating RepoEnv with Kubernetes backend...")
    
    env = RepoEnv(
        args=EnvArgs(ds[0]),
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