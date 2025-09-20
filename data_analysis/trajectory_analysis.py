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


def analyse(datasets: list[str]):
    pred_files_counts: list[int] = []
    pred_lines_counts: list[int] = []
    bug_files_counts: list[int] = []
    bug_lines_counts: list[int] = []
    repo_counts = defaultdict(lambda: 0)
    instance_counts = defaultdict(lambda: 0)
    traj_count = 0

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

            traj_count += 1
            instance_counts[traj['ds']['problem_statement']] += 1

            if "patch" in traj['ds']:
                bug_patch = traj['ds']['patch']
            elif "parsed_commit_content" in traj['ds']:
                commit = ParsedCommit(**json.loads(traj['ds']['parsed_commit_content']))
                bug_patch = commit.get_patch()
            else:
                raise RuntimeError(f"Count not find relevent key for patch in {traj['ds'].keys()}")

            bug_files_modified, bug_lines_changed = count_patch_stats(bug_patch)
            bug_files_counts.append(bug_files_modified)
            bug_lines_counts.append(bug_lines_changed)

            pred_patch = traj.get("output_patch", "") or ""
            pred_files_modified, pred_lines_changed = count_patch_stats(pred_patch)
            pred_files_counts.append(pred_files_modified)
            pred_lines_counts.append(pred_lines_changed)

            found_repo = False
            for possible_name in ["repo", "repo_name"]:
                if possible_name in traj['ds']:
                    repo_counts[traj['ds'][possible_name]] += 1
                    found_repo = True
                    break
            if not found_repo:
                print(f"Could not find repo name!")

    return {
        "pred_files_mean": mean(pred_files_counts),
        "pred_files_std": std(pred_files_counts),
        "pred_lines_mean": mean(pred_lines_counts),
        "pred_lines_std": std(pred_lines_counts),
        "count": traj_count,
        "bug_files_mean": mean(bug_files_counts),
        "bug_files_std": std(bug_files_counts),
        "bug_lines_mean": mean(bug_lines_counts),
        "bug_lines_std": std(bug_lines_counts),
        "repo_counts": repo_counts,
        # Sorted descending counts per instance
        "instance_counts": list(sorted(instance_counts.values(), reverse=True)),
    }


def main():
    console = Console()
    dataset_root = Path("/home/msrt/atharv/data/collected_trajectories")
    dataset_paths = {
        k: list(dataset_root.rglob(f"{k}/claude4/*.jsonl"))
        for k in ["d1", "d2", "d3", "d4"]
    }
    pprint(dataset_paths)

    overall_repo_counts = defaultdict(int)
    overall_instance_counts: list[int] = []

    per_dataset_repo_counts: dict[str, dict[str, int]] = {}
    per_dataset_instance_counts: dict[str, list[int]] = {}

    table = Table(title="Patch Stats by Dataset Group")
    table.add_column("Dataset", style="bold")
    table.add_column("count")
    table.add_column("pred_files")
    table.add_column("pred_lines")
    table.add_column("bug_files")
    table.add_column("bug_lines")

    for d_name, d_paths in dataset_paths.items():
        stats = analyse([str(p) for p in d_paths])
        table.add_row(
            d_name,
            f"{stats['count']}",
            f"{stats['pred_files_mean']:.2f}",
            f"{stats['pred_lines_std']:.2f}",
            f"{stats['bug_files_mean']:.2f}",
            f"{stats['bug_lines_std']:.2f}",
        )

        # collect per-dataset for dashboard
        repo_counts = dict(stats["repo_counts"])
        per_dataset_repo_counts[d_name] = repo_counts
        per_dataset_instance_counts[d_name] = list(stats["instance_counts"])

        # update overall
        for k, v in repo_counts.items():
            overall_repo_counts[k] += v
        overall_instance_counts.extend(stats["instance_counts"])

    console.print(table)
    
    # Ensure overall instance counts are also sorted for the series plot
    overall_instance_counts = sorted(overall_instance_counts, reverse=True)
    
    # # save one multi-plot PNG
    # dashboard_path = Path("patch_stats_dashboard.png")
    # save_dashboard(
    #     per_dataset_repo_counts=per_dataset_repo_counts,
    #     per_dataset_instance_counts=per_dataset_instance_counts,
    #     overall_repo_counts=dict(overall_repo_counts),
    #     overall_instance_counts=overall_instance_counts,
    #     out_path=dashboard_path,
    #     top_n_repos=10,
    # )

    # console.print(f"[green]Saved dashboard to:[/green] {dashboard_path.resolve()}")


if __name__ == "__main__":
    main()
