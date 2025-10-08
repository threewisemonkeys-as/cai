import json
import math
from pathlib import Path
from typing import Iterable
from collections import defaultdict

from unidiff import PatchSet
from rich.console import Console
from rich.table import Table
from rich import print
from rich.pretty import pprint

from r2egym.commit_models.diff_classes import ParsedCommit



def count_patch_stats(patch_str: str) -> tuple[int, int]:
    """
    Count number of files modified and number of lines changed
    (added + removed) from a unified diff string.
    
    Args:
        patch_str: String containing the unified diff.
    
    Returns:
        (files_modified, lines_modified)
    """
    patch = PatchSet(patch_str.splitlines(keepends=True))

    files_modified = [h for h in patch if h.is_modified_file]
    files_modified_count = 0
    lines_modified_count = 0
    for mf in files_modified:
        if not mf.path.endswith(".py") or "test_" in mf.path:
            continue
        files_modified_count += 1
        for hunk in mf:
            for line in hunk:
                if line.is_added or line.is_removed:
                    lines_modified_count += 1
    
    return files_modified_count, lines_modified_count



def analyse(datasets: list[str]):
    results = []

    for ds in datasets:
        print(f"Analysing {ds}")
        with open(ds, "r") as f:
            data = []
            for l in f.read().splitlines():
                try:
                    data.append(json.loads(l.strip()))
                except json.JSONDecodeError:
                    print("Skpping line due to decode error")

        for traj in data:
            if traj['reward'] != 1.0:
                continue

            if "patch" in traj['ds']:
                bug_patch = traj['ds']['patch']
            elif "parsed_commit_content" in traj['ds']:
                commit = ParsedCommit(**json.loads(traj['ds']['parsed_commit_content']))
                bug_patch = commit.get_patch()
            else:
                raise RuntimeError(f"Count not find relevent key for patch in {traj['ds'].keys()}")

            bug_files_modified, bug_lines_changed = count_patch_stats(bug_patch)

            pred_patch = traj.get("output_patch", "") or ""
            pred_files_modified, pred_lines_changed = count_patch_stats(pred_patch)

            results.append((traj, {
                "bfm": bug_files_modified,
                "blc": bug_lines_changed,
                "pfm": pred_files_modified,
                "plc": pred_lines_changed,
            }))


    return results

def main():
    dataset_root = "/home/msrt/atharv/data/trajectories_d4_sep21/"
    res = analyse([str(p) for p in Path(dataset_root).rglob("*.jsonl")])
    breakpoint()


if __name__ == "__main__":
    main()
