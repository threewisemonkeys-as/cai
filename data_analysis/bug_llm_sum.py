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
    cumsum_output_path = Path("/home/msrt/atharv/data/llm_cumsum.json")

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


if __name__ == "__main__":
    main()
