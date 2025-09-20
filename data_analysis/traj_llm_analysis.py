import json
from pathlib import Path
import random

from unidiff import PatchSet
from rich.console import Console
from rich import print
from rich.pretty import pprint
from litellm import completion

from r2egym.commit_models.diff_classes import ParsedCommit


MODEL = "openai/gpt-5-mini-2025-08-07"

def get_batches(seq: list[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def llm_summarise_conv(
    prompts: list[str],
    n: int = 2,
) -> str | None:
    assert n > 1, f"n should be strictly greater than 1, currently {n}"

    sequence = [prompts]

    while len(prompts) > 1:
        random.shuffle(prompts)
        new_prompts = []
        for batch in get_batches(prompts, n=n):

            if len(batch) == 1:
                new_prompts.append(batch[0])
            else:
                batch_str = ''
                for i, b in enumerate(batch):
                    batch_str += f'<idx_{i}>\n{b}\n</idx_{i}>\n\n'
                summary_prompt = f"Provide a consice summary of the following -\n\n{batch_str}"
                llm_repsonse = completion(
                    model=MODEL,
                    messages=[{"role": "user", "content": summary_prompt}],
                )
                summary = llm_repsonse["choices"][0]["message"]["content"]
                if summary is not None:
                    new_prompts.append(summary)
        
        sequence.append(new_prompts)
        prompts = new_prompts

    return sequence



def get_non_test_patch(patch: str) -> str:
    return str(PatchSet([
        str(f) for f in PatchSet(patch)
        if f.is_modified_file and f.path.endswith(".py") and not "test_" in f.path
    ]))
    


def llm_get_difference(traj) -> str | None:
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

    prompt = f"""Your task is to summarise in brief the differences between the two ways to resolve a bug in a codebase.
Especially consider if they are significantly different in size.
Only consider the differences between the solutions, do not consider any other possible approaches to the problem.
The bug is described below - 
<problem_statement>
{ps}
</problem_statement>

Solution A:
<solution_a>
{gt_patch}
</solution_a>

Solution B:
<solution_b>
{pred_patch}
</solution_b>
"""
    llm_repsonse = completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    difference = llm_repsonse["choices"][0]["message"]["content"]

    return difference


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

    differences = []
    for traj in trajs_to_eval:
        diff = llm_get_difference(traj)
        if diff is not None: 
            differences.append(diff)
        
        if len(differences) >= sample_n:
            break
    
    return differences


def main():
    dataset_root = Path("/home/msrt/atharv/data/collected_trajectories")
    diff_output_path = Path("/home/msrt/atharv/data/llm_diff.json")
    sumseq_output_path = Path("/home/msrt/atharv/data/llm_sumseq.json")

    console = Console()
    dataset_paths = {
        k: list(dataset_root.rglob(f"{k}/claude4/*.jsonl"))
        for k in ["d1", "d2", "d3", "d4"]
    }
    pprint(dataset_paths)


    all_differences = {}
    for d_name, d_paths in dataset_paths.items():
        d_differences = analyse([str(p) for p in d_paths], sample_n=64)
        all_differences[d_name] = d_differences

    print(f"Writing differences to {diff_output_path}")
    json.dump(all_differences, open(diff_output_path, "w"), indent=4)

    all_differences = json.load(open(diff_output_path, "r"))

    all_sumseqs = {}
    for d_name, d_differences in all_differences.items():
        d_sumseq = llm_summarise_conv(d_differences, n=8)
        all_sumseqs[d_name] = d_sumseq

    print(f"Writing sumseqs to {sumseq_output_path}")
    json.dump(all_sumseqs, open(sumseq_output_path, "w"), indent=4)
    

if __name__ == "__main__":
    main()
