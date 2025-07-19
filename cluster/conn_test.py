import os

from datasets import load_dataset

from r2egym.agenthub.runtime.docker import DockerRuntime

remote = None
try:
    os.environ["DOCKER_HOST"] = "ssh://dockerd-remote"
    print(f"DOCKER_HOST set to {os.environ.get('DOCKER_HOST', None)}")
    print("Loading dataset R2E-Gym/R2E-Gym-Subset (split=train)")
    ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
    print(f"Dataset loaded: {len(ds)} items, first keys: {list(ds[0].keys())}")
    print("Starting DockerRuntime …")
    remote = DockerRuntime(
        ds=ds[-1],
        command=["/bin/bash", "-l"],
        backend="docker",
    )
    print("DockerRuntime started")
    cmd = "ls /testbed"
    print(f"Executing in container: {cmd}")
    result = remote.run(cmd)
    print(f"Command completed: {result}")

finally:
    print("Closing DockerRuntime …")
    if remote is not None:
        remote.close()
    print("DockerRuntime closed")

