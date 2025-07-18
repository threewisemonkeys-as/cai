import os

os.environ["DOCKER_HOST"] = "ssh://dockerd-remote2"

import docker
from datasets import load_dataset
from r2egym.agenthub.runtime.docker import DockerRuntime

ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
remote = DockerRuntime(ds=ds[0], command=["/bin/bash", "-l"], backend="docker")
result = remote.run("ls /testbed")
remote.close()
print(result)
