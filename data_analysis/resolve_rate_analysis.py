import json
import numpy as np
import matplotlib.pyplot as plt  # changed from previous solution

from unidiff import PatchSet
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

    files_modified = [h for h in patch]
    lines_modified_count = 0
    for mf in files_modified:
        for hunk in mf:
            for line in hunk:
                if line.is_added or line.is_removed:
                    lines_modified_count += 1
    
    return len(files_modified), lines_modified_count


def main():

    rr_data_paths = [
        "/home/msrt/atharv/data/claude4_task_difficulty_level/d1_task_difficulty_level.jsonl",
        "/home/msrt/atharv/data/claude4_task_difficulty_level/d2_task_difficulty_level.jsonl",
        "/home/msrt/atharv/data/claude4_task_difficulty_level/d3_task_difficulty_level.jsonl",
        "/home/msrt/atharv/data/claude4_task_difficulty_level/d4_task_difficulty_level.jsonl",
    ]

    bugs_data_paths = [
        "/home/msrt/data/rl_tasks/r2egym_train.json",
        "/home/msrt/data/rl_tasks/swesmith_train.json",
        "/home/msrt/data/rl_tasks/buggen_train.json",
        "/home/msrt/data/rl_tasks/featadd_train.json",
    ]

    results = {}
    for rr_path, bugs_data_path in zip(rr_data_paths, bugs_data_paths):
        rr_data = [json.loads(l) for l in open(rr_path, "r").read().splitlines()]
        bugs_data = json.load(open(bugs_data_path, "r"))
        
        bugs_map = {
            d["problem_statement"][:100]: d for d in bugs_data
            if "problem_statement" in d and d["problem_statement"] is not None
        }
        points = []
        for problem in rr_data:
            success_rate = problem["success_rate"]
            if problem["problem_statement_prefix"] not in bugs_map:
                print(f"Skipping")
                continue

            bug = bugs_map[problem["problem_statement_prefix"]]

            if "patch" in bug:
                patch = bug['patch']
            elif "parsed_commit_content" in bug:
                commit = ParsedCommit(**json.loads(bug['parsed_commit_content']))
                patch = commit.get_patch()
            else:
                raise RuntimeError(f"Count not find relevent key for patch in {bug.keys()}")

            files_changes, lines_changes = count_patch_stats(patch)
            points.append((success_rate, files_changes, lines_changes))
        results[bugs_data_path] = points

    # ---------- heat map plotting block ----------
    # Gather all points to compute shared binning
    all_points = [p for ds in bugs_data_paths for p in results.get(ds, [])]
    if not all_points:
        print("No data to plot.")
        return

    # Resolve rate bins (fixed 0..1)
    x_bins = np.linspace(0.0, 1.0, 21)

    # Robust caps for files/lines to avoid long tails dominating
    files_all = np.array([p[1] for p in all_points], dtype=float)
    lines_all = np.array([p[2] for p in all_points], dtype=float)

    files_cap = max(1, int(np.percentile(files_all, 99)))
    lines_cap = max(1, int(np.percentile(lines_all, 99)))

    y_files_bins = np.arange(0, files_cap + 2)  # integer-ish bins
    y_lines_bins = np.linspace(0, lines_cap, min(lines_cap + 1, 60)) if lines_cap > 0 else np.array([0, 1])

    def plot_heat(ax, x_vals, y_vals, x_edges, y_edges, title, xlab, ylab):
        if len(x_vals) == 0:
            ax.set_title(title)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return None
        H, xe, ye = np.histogram2d(x_vals, y_vals, bins=[x_edges, y_edges])
        # Show transposed with origin lower so y increases upward
        im = ax.imshow(
            H.T,
            origin="lower",
            aspect="auto",
            extent=[xe[0], xe[-1], ye[0], ye[-1]],
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        return im

    fig, axes = plt.subplots(5, 2, figsize=(12, 18), constrained_layout=True)

    imgs_col0 = []
    imgs_col1 = []

    # First 4 rows: per dataset
    for row_idx, ds_path in enumerate(bugs_data_paths):
        pts = results.get(ds_path, [])
        x = np.array([p[0] for p in pts], dtype=float)
        y_files = np.array([p[1] for p in pts], dtype=float)
        y_lines = np.array([p[2] for p in pts], dtype=float)

        im0 = plot_heat(
            axes[row_idx, 0], x, np.clip(y_files, 0, files_cap),
            x_bins, y_files_bins,
            f"{ds_path} — Resolve vs Files",
            "Resolve rate", "# Files changed"
        )
        im1 = plot_heat(
            axes[row_idx, 1], x, np.clip(y_lines, 0, lines_cap),
            x_bins, y_lines_bins,
            f"{ds_path} — Resolve vs Lines",
            "Resolve rate", "# Lines changed"
        )
        if im0 is not None: imgs_col0.append(im0)
        if im1 is not None: imgs_col1.append(im1)

    # Last row: combined
    x_all = np.array([p[0] for p in all_points], dtype=float)
    files_c = np.clip(files_all, 0, files_cap)
    lines_c = np.clip(lines_all, 0, lines_cap)

    im0 = plot_heat(
        axes[4, 0], x_all, files_c,
        x_bins, y_files_bins,
        "Combined — Resolve vs Files",
        "Resolve rate", "# Files changed"
    )
    im1 = plot_heat(
        axes[4, 1], x_all, lines_c,
        x_bins, y_lines_bins,
        "Combined — Resolve vs Lines",
        "Resolve rate", "# Lines changed"
    )
    if im0 is not None: imgs_col0.append(im0)
    if im1 is not None: imgs_col1.append(im1)

    # Shared colorbars per column
    if imgs_col0:
        fig.colorbar(imgs_col0[-1], ax=axes[:, 0], label="Count")
    if imgs_col1:
        fig.colorbar(imgs_col1[-1], ax=axes[:, 1], label="Count")

    plt.savefig("resolve_vs_changes_heatmaps.png", dpi=200)
    # plt.show()
    # ---------- end plotting block ----------


if __name__ == '__main__':
    main()
