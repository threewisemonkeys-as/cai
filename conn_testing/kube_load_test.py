#!/usr/bin/env python
"""
k8s_load_test_fire.py

Stress-test a Kubernetes cluster by spawning many concurrent
DockerRuntime (backend="kubernetes") sessions and running a shell
command in each pod.

Example:
    # 20 parallel pods, 50 random dataset rows, 10 s timeout per pod
    python k8s_load_test_fire.py run --workers 20 --samples 50 \
        --kubeconfig ~/.kube/k3s.yaml --cmd "python -V" --timeout 10
"""
from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Tuple

import fire
from datasets import load_dataset
from r2egym.agenthub.runtime.docker import DockerRuntime

DEFAULT_TIMEOUT = 30


def _run_one(sample: Any, cmd: str, timeout: int) -> Tuple[str, bool, float, str | None, str | None]:
    """Executes *cmd* inside a fresh pod for *sample* and returns result tuple."""
    start = time.perf_counter()
    rt = None
    try:
        rt = DockerRuntime(
            ds=sample,
            command=["/bin/bash", "-l"],
            backend="kubernetes",
        )
        out, exit_status = rt.run(cmd, timeout=timeout)
        ok = (exit_status == "0")
        return sample.get("docker_image", "unknown_instance"), ok, time.perf_counter() - start, (
            None if ok else out
        ), out
    except Exception as exc:  # noqa: BLE001
        return sample.get("docker_image", "unknown_instance"), False, time.perf_counter() - start, str(exc), None
    finally:
        if rt is not None:
            try:
                rt.stop_container()
            except Exception:  # noqa: BLE001
                pass


def run(
    workers: int = 10,
    samples: int | None = None,
    kubeconfig: str | None = None,
    cmd: str = "ls /testbed",
    timeout: int = DEFAULT_TIMEOUT,
) -> None:
    """
    Launch *workers* concurrent pods picked from *samples* random dataset entries.

    Args:
        workers: Number of concurrent pods to run.
        samples: Dataset rows to test (random pick).  Defaults to *workers*.
        kubeconfig: Path to kube-config; if omitted uses $KUBECONFIG or in-cluster.
        cmd: Shell command executed in each pod.
        timeout: Seconds before the command inside a pod is killed.
    """
    # Configure client-side kube access
    if kubeconfig:
        os.environ["KUBECONFIG"] = str(Path(kubeconfig).expanduser())
    print(f"KUBECONFIG={os.environ.get('KUBECONFIG', '(default resolution)')}")

    print("Loading dataset R2E-Gym/R2E-Gym-Subset (split=train)…")
    ds = load_dataset("R2E-Gym/R2E-Gym-Subset", split="train")
    print(f"Dataset loaded: {len(ds)} items")

    n_samples = samples or workers
    n_samples = min(n_samples, len(ds))
    chosen = [ds[i] for i in random.sample(range(len(ds)), k=n_samples)]

    print(f"Launching {workers} worker(s) over {n_samples} sample(s)…")
    successes = failures = 0
    total_time = 0.0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, s, cmd, timeout): s for s in chosen}
        for fut in as_completed(futures):
            idx, ok, elapsed, err, out = fut.result()
            total_time += elapsed
            label = "OK " if ok else "ERR"
            print(f"[{label}] task={idx:<8}  t={elapsed:6.2f}s  {err or out.strip()}")
            successes += ok
            failures += (not ok)

    total = successes + failures
    avg = total_time / total if total else 0.0
    print(
        f"\nSummary: {successes} succeeded / {failures} failed "
        f"({total} total); avg latency = {avg:.2f}s"
    )


if __name__ == "__main__":
    fire.Fire(run)
