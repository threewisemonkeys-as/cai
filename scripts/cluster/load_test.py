"""
docker_load_test.py

Stress‑test a remote Docker daemon by opening many simultaneous
DockerRuntime sessions and executing an inexpensive command inside each.

Usage:
    python docker_load_test.py --workers 20 --samples 50 \
        --docker-host ssh://dockerd-remote
"""
import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from r2egym.agenthub.runtime.docker import DockerRuntime


def _run_one(sample, cmd):
    """
    Helper executed in a worker thread.

    Returns (idx, success, elapsed_sec, stderr_or_none)
    """
    start = time.perf_counter()
    remote = None
    try:
        remote = DockerRuntime(ds=sample, command=["/bin/bash", "-l"], backend="docker")
        out = remote.run(cmd)
        elapsed = time.perf_counter() - start
        return sample["docker_image"], True, elapsed, None, out
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        return sample.get("docker_image", "unknown_instance"), False, elapsed, str(exc), None
    finally:
        if remote is not None:
            remote.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10, help="Concurrent sessions")
    parser.add_argument("--samples", type=int, default=None, help="How many dataset items to test")
    parser.add_argument(
        "--docker-host",
        type=str,
        default=None,
        help="Value to export as DOCKER_HOST (e.g. ssh://dockerd-remote)",
    )
    parser.add_argument("--cmd", type=str, default="ls /testbed", help="Shell command to run")
    args = parser.parse_args()

    if args.docker_host:
        os.environ["DOCKER_HOST"] = args.docker_host
    print(f"DOCKER_HOST={os.environ.get('DOCKER_HOST', None)}")

    print("Loading dataset R2E-Gym/R2E-Gym-Subset (split=train)…")
    ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
    print(f"Dataset loaded: {len(ds)} items")

    # Pick 'samples' random rows to avoid hammering the same task image repeatedly
    n_samples = args.samples
    if n_samples is None:
        n_samples = args.workers
    if n_samples > len(ds):
        n_samples = len(ds)

    indices = random.sample(range(len(ds)), k=min(n_samples, len(ds)))
    samples = [ds[i] for i in indices]

    print(f"Launching {args.workers} worker(s) over {len(samples)} sample(s)…")
    successes, failures, total_time = 0, 0, 0.0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, s, args.cmd): s for s in samples}
        for fut in as_completed(futures):
            idx, ok, elapsed, err, out = fut.result()
            total_time += elapsed
            label = "OK " if ok else "ERR"
            print(f"[{label}] task={idx:<8}  t={elapsed:6.2f}s  {err or out}")
            if ok:
                successes += 1
            else:
                failures += 1

    print(
        f"\nSummary: {successes} succeeded / {failures} failed "
        f"({successes+failures} total); "
        f"avg latency = {total_time/(successes+failures):.2f}s"
    )


if __name__ == "__main__":
    main()
