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
    per_patch_sizes = []
    per_files_modified = []
    f2p_tests = []

    for dp in data:
        if "patch" in dp:
            patch = dp['patch']
        elif "parsed_commit_content" in dp:
            commit = ParsedCommit(**json.loads(dp['parsed_commit_content']))
            patch = commit.get_patch()
        else:
            raise RuntimeError(f"Count not find relevent key for patch in {dp.keys()}")
        
        # patch = get_non_test_patch(patch)
        fm, lm = count_patch_stats(patch)
        # fm, lm = analyze_patch(patch)
        per_files_modified.append(fm)
        per_patch_sizes.append(lm)
        
        if "FAIL_TO_PASS" in dp:
            f2p_tests.append(len(dp["FAIL_TO_PASS"]))
        elif "num_non_test_func_methods" in dp and "modified_entity_summaries" in dp:
            total_func_methods = get_total_func_methods_r2eg(dp)
            if dp["num_non_test_func_methods"] > total_func_methods:
                raise RuntimeError(f"Number of functions found is less than number of non test functions in dataset")
            f2p_tests.append(total_func_methods - dp["num_non_test_func_methods"])
        else:
            raise RuntimeError("Unknown data format!")

    return per_patch_sizes, per_files_modified, f2p_tests, len(data)


def _mean_std(values):
    """Return (mean, std). Uses population std by default."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(var)
    return (mean, std)


def main():

    data_paths = [
        "/home/msrt/data/rl_tasks/r2egym_train.json",
        "/home/msrt/data/rl_tasks/swesmith_train.json",
        "/home/msrt/data/rl_tasks/buggen_train.json",
        "/home/msrt/data/rl_tasks/featadd_train.json",
    ]


    console = Console()
    table = Table(title="Patch Statistics Across Datasets")
    table.add_column("Dataset", style="bold")
    table.add_column("n", justify="right")
    table.add_column("Patch Size (mean)", justify="right")
    table.add_column("Patch Size (std)", justify="right")
    table.add_column("Files Modified (mean)", justify="right")
    table.add_column("Files Modified (std)", justify="right")
    table.add_column("F2P Tests (mean)", justify="right")
    table.add_column("F2P Tests (std)", justify="right")

    for data_path in data_paths:
        data_path = Path(data_path)
        patch_sizes, files_modified, f2p_tests, n = collect_stats(data_path)
        ps_mean, ps_std = _mean_std(patch_sizes)
        fm_mean, fm_std = _mean_std(files_modified)
        f2p_mean, f2p_std = _mean_std(f2p_tests)

        table.add_row(
            data_path.name,
            str(n),
            f"{ps_mean:.3f}",
            f"{ps_std:.3f}",
            f"{fm_mean:.3f}",
            f"{fm_std:.3f}",
            f"{f2p_mean:.3f}",
            f"{f2p_std:.3f}",
        )

    console.print(table)



if __name__ == '__main__':
    main()