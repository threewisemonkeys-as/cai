import json
from pathlib import Path
import random

from unidiff import PatchSet
from rich.console import Console
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


def get_lens(traj):
    ps = traj['ds']['problem_statement']
    
    if "patch" in traj['ds']:
        gt_patch = traj['ds']['patch']
    elif "parsed_commit_content" in traj['ds']:
        commit = ParsedCommit(**json.loads(traj['ds']['parsed_commit_content']))
        gt_patch = commit.get_patch()
    else:
        raise RuntimeError(f"Count not find relevent key for patch in {traj['ds'].keys()}")

    pred_patch = traj['output_patch']

    gt_patch, pred_patch = get_non_test_patch(gt_patch), get_non_test_patch(pred_patch)

    if gt_patch is None or gt_patch.strip() == '' or pred_patch is None or pred_patch == '':
        return None
    
    gt_fm, gt_lm = count_patch_stats(gt_patch)
    p_fm, p_lm = count_patch_stats(pred_patch)

    return gt_lm, p_lm
    



def get_non_test_patch(patch: str) -> str:
    return str(PatchSet([
        str(f) for f in PatchSet(patch)
        if f.is_modified_file and f.path.endswith(".py") and not "test_" in f.path
    ]))

def analyse(datasets: list[str], sample_n: int | None = None):
    trajs_to_eval = []

    for ds in datasets:
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
        
            trajs_to_eval.append(traj)


    random.shuffle(trajs_to_eval)

    gt_len, pred_len = [], []
    for traj in trajs_to_eval:
        r = get_lens(traj)
        if r is not None:
            gr, pr = r
            gt_len.append(gr)
            pred_len.append(pr)

    return sum(gt_len) / len(gt_len), sum(pred_len) / len(pred_len)


def main():
    dataset_root = Path("/home/msrt/atharv/data/collected_trajectories")

    dataset_paths = {
        k: list(dataset_root.rglob(f"{k}/claude4/*.jsonl"))
        for k in ["d1", "d2", "d3", "d4"]
    }
    pprint(dataset_paths)


    for d_name, d_paths in dataset_paths.items():
        gr, pr = analyse([str(p) for p in d_paths], sample_n=64)
        print(f"{d_name}\t\t{gr}\t\t{pr}")

if __name__ == "__main__":
    main()
