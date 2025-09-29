import json
import math
from pathlib import Path
from typing import Iterable

from unidiff import PatchSet
from rich.console import Console
from rich.table import Table
from rich import print


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

    
    # files_modified = [h for h in patch if h.is_modified_file]
    files_modified = [h for h in patch]
    lines_modified_count = 0
    for mf in files_modified:
        for hunk in mf:
            for line in hunk:
                if line.is_added or line.is_removed:
                    lines_modified_count += 1
    
    return len(files_modified), lines_modified_count



def analyze_patch(patch_str: str):
    added, removed = 0, 0
    files = set()

    for line in patch_str.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 3:
                files.add(parts[2][2:])  # strip "a/"
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1

    size = added + removed
    return len(files), size


def get_non_test_patch(patch: str) -> str:
    return str(PatchSet([
        str(f) for f in PatchSet(patch)
        if f.is_modified_file and f.path.endswith(".py") and not "test_" in f.path
    ]))
    


def get_total_func_methods_r2eg(entry: dict) -> int:
    count = 0
    for entity in entry["modified_entity_summaries"]:
        if entity["type"] in ["method", "function"]:
            count += 1
    return count


def collect_stats(data_path: Path):
    data = json.load(open(data_path, "r"))
    results = []

    for dp in data:
        if "patch" in dp:
            patch = dp['patch']
        elif "parsed_commit_content" in dp:
            commit = ParsedCommit(**json.loads(dp['parsed_commit_content']))
            patch = commit.get_patch()
        else:
            raise RuntimeError(f"Count not find relevent key for patch in {dp.keys()}")
        
        patch = get_non_test_patch(patch)
        fm, lm = count_patch_stats(patch)
        # fm, lm = analyze_patch(patch)

        results.append(
            (dp, ({
                "fm": fm,
                "lm": lm,
            }))
        )

    return results


def main():

    data_paths = [
        # "/home/msrt/atharv/data/clean_featadd_train.json",
        # "/home/msrt/data/rl_tasks/r2egym_train.json",
        "/home/msrt/data/rl_tasks/swesmith_train.json",
        # "/home/msrt/data/rl_tasks/buggen_train.json",
        # "/home/msrt/data/rl_tasks/featadd_train.json",
    ]


    for data_path in data_paths:
        data_path = Path(data_path)
        res = collect_stats(data_path)
        breakpoint()




if __name__ == '__main__':
    main()