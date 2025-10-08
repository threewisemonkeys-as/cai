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


def _barh_top_n(ax, counter_dict: dict[str, int], title: str, top_n: int = 10):
    import matplotlib.pyplot as plt
    items = sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not items:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    labels, values = zip(*items)
    y = range(len(values))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Count")


def _hist(ax, values: list[int], title: str, bins="auto"):
    import matplotlib.pyplot as plt
    if not values:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Count per instance")
    ax.set_ylabel("Frequency")


def _series_sorted_plot(ax, values: list[int], title: str):
    """
    Plot a simple series of counts (sorted descending) for all instances.
    X-axis is just the instance index (no instance labels).
    """
    import matplotlib.pyplot as plt
    if not values:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    vals = sorted(values, reverse=True)
    x = range(1, len(vals) + 1)
    ax.plot(x, vals, marker='.', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Instances (sorted)")
    ax.set_ylabel("# Trajectories")
    if len(vals) > 20:
        ax.set_xticks([])


def _overlay_series(ax, series_by_ds: dict[str, list[int]], ds_order: list[str], title: str):
    """
    Overlay the per-dataset 'trajs per instance' (each sorted desc) in one panel.
    """
    import matplotlib.pyplot as plt
    plotted = False
    for d in ds_order:
        vals = series_by_ds.get(d, [])
        if not vals:
            continue
        vals = sorted(vals, reverse=True)
        x = range(1, len(vals) + 1)
        ax.plot(x, vals, marker='.', linewidth=1, label=d)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("Instances (sorted per dataset)")
    ax.set_ylabel("# Trajectories")
    if plotted:
        ax.legend(fontsize=8, frameon=False)
        # keep x ticks minimal for large N across datasets
        ax.set_xticks([])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])


def save_dashboard(
    per_dataset_repo_counts: dict[str, dict[str, int]],
    per_dataset_instance_counts: dict[str, list[int]],
    overall_repo_counts: dict[str, int],
    overall_instance_counts: list[int],
    out_path: Path,
    top_n_repos: int = 10,
):
    """
    Create a single PNG with multiple plots:
      Row 1: barh of top-N repos per dataset + 'All' (+ blank overlay col)
      Row 2: histogram of instance_counts per dataset + 'All' (+ blank overlay col)
      Row 3: trajectories-per-instance series (sorted desc) per dataset + 'All' + OVERLAY (all datasets)
    """
    import matplotlib.pyplot as plt

    datasets = list(per_dataset_repo_counts.keys())
    # +1 col for "All" and +1 extra final col for the overlay panel
    cols = len(datasets) + 2
    fig, axes = plt.subplots(nrows=3, ncols=cols, figsize=(5 * cols, 9))

    # Ensure axes is 2D indexable when cols == 1 (unlikely here, but safe)
    if cols == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]

    all_idx = len(datasets)        # column for "All"
    overlay_idx = len(datasets) + 1  # final column for overlay

    # Row 1: barh repos
    for i, d in enumerate(datasets):
        _barh_top_n(axes[0][i], per_dataset_repo_counts[d], f"Top repos — {d}", top_n=top_n_repos)
    _barh_top_n(axes[0][all_idx], overall_repo_counts, "Top repos — All", top_n=top_n_repos)
    # hide the extra overlay column in row 1
    axes[0][overlay_idx].axis("off")

    # Row 2: hist instance counts
    for i, d in enumerate(datasets):
        _hist(axes[1][i], per_dataset_instance_counts[d], f"Instance counts — {d}", bins="auto")
    _hist(axes[1][all_idx], overall_instance_counts, "Instance counts — All", bins="auto")
    # hide the extra overlay column in row 2
    axes[1][overlay_idx].axis("off")

    # Row 3: trajectories per instance (sorted)
    for i, d in enumerate(datasets):
        _series_sorted_plot(axes[2][i], per_dataset_instance_counts[d], f"Trajs per instance — {d}")
    _series_sorted_plot(axes[2][all_idx], overall_instance_counts, "Trajs per instance — All")
    # NEW: Overlay of all datasets
    _overlay_series(axes[2][overlay_idx], per_dataset_instance_counts, datasets, "Overlay — Trajs per instance")

    fig.suptitle("Patch Stats Dashboard", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def mean(data: Iterable[float]) -> float | None:
    data = list(data)
    if not data:
        return None
    return sum(data) / len(data)


def std(data: Iterable[float]) -> float | None:
    """
    Population standard deviation (no Bessel's correction).
    Returns None for empty, and 0.0 for length==1 (no variability).
    """
    data = list(data)
    if not data:
        return None
    if len(data) == 1:
        return 0.0
    mu = mean(data)
    assert mu is not None
    var = sum((x - mu) ** 2 for x in data) / len(data)
    return math.sqrt(var)


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

def get_patch_file_overlap(
    *patches
) -> list[str]:
    modified_files = []
    for p in patches:
        modified_files.append(set([
            f.path for f in PatchSet(p)
            if f.is_modified_file and f.path.endswith(".py") and not "test" in f.path
        ]))
    return list(set.intersection(*modified_files))


def get_first_hit(
    steps: list[dict],
    patch: str,
) -> int | None:
    modified_files = [
        f.path for f in PatchSet(patch) 
        if f.is_modified_file and f.path.endswith(".py") and not "test" in f.path
    ]
    
    for idx, step in enumerate(steps):
        for mf in modified_files:
            if mf in step['action']:
                return idx
            
    return None



def analyse(
        datasets: list[str],
        bug_types: list[str],
        bug_data: dict[str, str]
    ):
    traj_count = 0
    bug_type_result = {
        k: defaultdict(lambda: 0) for k in bug_types
    }

    bug_type_map = {
        bug['problem_statement']: btype
        for bug, btype in bug_data
    }
 
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

            if traj['ds']['problem_statement'] not in bug_type_map:
                continue

            curr_btype = bug_type_map[traj['ds']['problem_statement']]
            
            if curr_btype not in bug_type_result:
                continue

            traj_count += 1
            bug_type_result[curr_btype]["count"] += 1
            bug_type_result[curr_btype]["reward"] += traj["reward"]

            
            



    return bug_type_result

def main():
    console = Console()
    dataset_root = Path("/home/msrt/atharv/data/collected_trajectories")
    dataset_paths = {
        k: list(dataset_root.rglob(f"{k}/claude4/*.jsonl"))
        for k in ["d1", "d2", "d3", "d4"]
    }
    pprint(dataset_paths)


    bug_type_data_path = "/home/msrt/atharv/data/full_bug_type_results.json"
    bug_type_data = json.load(open(bug_type_data_path, "r"))

    bug_type_path_map = {
        "d1": '/home/msrt/data/rl_tasks/r2egym_train.json',
        "d2": '/home/msrt/data/rl_tasks/swesmith_train.json',
        "d3": '/home/msrt/data/rl_tasks/buggen_train.json',
        "d4": '/home/msrt/data/rl_tasks/featadd_train.json',
    }
    bug_types_to_include = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    results = {}
    for d_name, d_paths in dataset_paths.items():

        bug_data = bug_type_data[bug_type_path_map[d_name]]
        stats = analyse([str(p) for p in d_paths], bug_types_to_include, bug_data)
        results[d_name] = stats
        print(stats)


    print(results)

if __name__ == "__main__":
    main()
