import json
from pathlib import Path
import random


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
                summary_prompt = f"""Provide a consice summary of the following.
Attempt to generalise characteristics such as size, type and difficulty from the provided instance.

{batch_str}"""
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





def analyse(data: list[dict], sample_n: int | None = None, k: int = 2):
    base_summaries = []
    for bug in random.sample(data, sample_n):   
        if "patch" in bug:
            patch = bug['patch']
        elif "parsed_commit_content" in bug:
            commit = ParsedCommit(**json.loads(bug['parsed_commit_content']))
            patch = commit.get_patch()
        else:        
            raise RuntimeError(f"Count not find relevent key for patch in {dp.keys()}")
        ps = bug['problem_statement']
        prompt = f"""Your task is to summarise in brief the bug presented here.

<problem_description>
{ps}
</problem_description>

<patch>
{patch}
</patch>

Considering the given problem description and patch, summarise the bug in brief, including characteristics such as size, difficulty and type of bug."""
        llm_repsonse = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        bs = llm_repsonse["choices"][0]["message"]["content"]

        base_summaries.append(bs)

    results = llm_summarise_conv(base_summaries, k)    

    return results


def main():
    dataset_paths = [
        "/home/msrt/data/rl_tasks/r2egym_train.json",
        "/home/msrt/data/rl_tasks/swesmith_train.json",
        "/home/msrt/data/rl_tasks/buggen_train.json",
        "/home/msrt/data/rl_tasks/featadd_train.json",
    ]

    cumsum_output_path = Path("/home/msrt/atharv/data/llm_cumsum.json")

    pprint(dataset_paths)

    all_sums = {}
    for d_path in dataset_paths:
        data = json.load(open(d_path, "r"))
        d_sums = analyse(data, sample_n=64, k=4)
        all_sums[d_path] = d_sums

    print(f"Writing summaries to {cumsum_output_path}")
    json.dump(all_sums, open(cumsum_output_path, "w"), indent=4)


if __name__ == "__main__":
    main()
