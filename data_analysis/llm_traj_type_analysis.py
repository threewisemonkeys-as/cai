import json
from pathlib import Path
import random
from collections import defaultdict
import os

from rich import print
from rich.pretty import pprint
from litellm import completion
from tqdm import tqdm

from r2egym.commit_models.diff_classes import ParsedCommit

from dotenv import load_dotenv
load_dotenv()

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

AZURE_AD_TOKEN_PROVIDER = get_bearer_token_provider(DefaultAzureCredential(), os.getenv("AZURE_API_SCOPE", None))

MODEL = "azure/gpt-5_2025-08-07"
# MODEL = "openai/gpt-5-mini-2025-08-07"


def summarize_trajectory(traj: dict, max_obs_len: int = 500) -> str:
    """
    Summarize the sequence of actions and observations from a trajectory.

    Args:
        traj (dict): A trajectory like d[0].
        max_obs_len (int): Maximum length for an observation string. If longer,
                           truncate the middle.

    Returns:
        str: A formatted summary of the trajectory.
    """
    steps = traj.get("trajectory_steps", [])
    summary_lines = []

    for step in steps:
        action = step.get("action", "").strip()
        obs = step.get("observation", "").strip()

        if len(obs) > max_obs_len:
            half = max_obs_len // 2
            obs = obs[:half] + " ...[TRUNCATED]... " + obs[-half:]

        summary_lines.append(f"Step {step.get('step_idx', '?')}:")
        summary_lines.append(f"  Action: {action}")
        summary_lines.append(f"  Observation: {obs}")
        summary_lines.append("")  # blank line between steps

    return "\n".join(summary_lines)



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
                summary_prompt = f"""Generalise possible types from the provided summaries of bug-fixing trajectories.

{batch_str}"""
                llm_repsonse = completion(
                    model=MODEL,
                    messages=[{"role": "user", "content": summary_prompt}],
                    azure_ad_token_provider=AZURE_AD_TOKEN_PROVIDER,
                )
                summary = llm_repsonse["choices"][0]["message"]["content"]
                if summary is not None:
                    new_prompts.append(summary)
        
        sequence.append(new_prompts)
        prompts = new_prompts

    return sequence





def analyse(data: list[dict], sample_n: int | None = None, k: int = 2):
    base_summaries = []
    for traj in tqdm(random.sample(data, sample_n)):
        bug = traj['ds']
        ps = bug['problem_statement']
        traj_desc = summarize_trajectory(traj)
        output_patch = traj["output_patch"]

        prompt = f"""Your task is to summarise in brief the a bug-fixing trajectory presented here.

Problem that the trajectory is attempting to solve -
        
<problem_description>
{ps}
</problem_description>

Sequence of actions and observations in the trajectory -

<trajectory>
{traj_desc}
</trajectory>

Output patch produced by the trajectory -

<output_patch>
{output_patch}
</output_patch>

Considering the given problem description, trajectory and patch, summarise the trajectory in brief, including characteristics of the type of approach taken"""
        llm_repsonse = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            azure_ad_token_provider=AZURE_AD_TOKEN_PROVIDER,
        )
        bs = llm_repsonse["choices"][0]["message"]["content"]

        base_summaries.append(bs)

    results = llm_summarise_conv(base_summaries, k)    

    return results



def categorise(guide: str, data: list[dict], sample_n: int | None = None):
    results = []
    if sample_n is not None:
        samples = random.sample(data, sample_n)
    else:
        samples = data
    for traj in tqdm(samples):
        bug = traj['ds']
        ps = bug['problem_statement']
        traj_desc = summarize_trajectory(traj)
        output_patch = traj["output_patch"]
        prompt = f"""Your task is to categorise a bug-fixing trajecoty into a set of given types.

Here are the guidelines on the types - 
{guide}

Problem that the trajectory is attempting to solve -
        
<problem_description>
{ps}
</problem_description>

Sequence of actions and observations in the trajectory -

<trajectory>
{traj_desc}
</trajectory>

Output patch produced by the trajectory -

<output_patch>
{output_patch}
</output_patch>

Your response should be in xml format:
<reasoning>
Thinking about which categories that the given trajectory falls into.
</reasoning>
<category>
Alphabet code of category that trajectory falls into.
</category>
"""
        llm_repsonse = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            azure_ad_token_provider=AZURE_AD_TOKEN_PROVIDER,
        )
        bs = llm_repsonse["choices"][0]["message"]["content"]

        if '<category>' in bs and '</category>' in bs.split('<category>')[1]:
            cat = bs.split('<category>')[1].split('</category>')[0].strip()
            results.append((bug, cat))
        else:
            continue


    return results



def main():
    dataset_paths = [
        "/home/v-asonwane/cai/data/rllmlogs-full/R2EGym-32B-Agent-d12-d4-sep21/R2EGym-32B-Agent-d12-d4-sep21_seed0.jsonl"
    ]

    cumsum_output_path = Path("/home/v-asonwane/cai/data/traj_type/llm_trajtype.json")
    summary_output_path = Path("/home/v-asonwane/cai/data/traj_type/traj_types.txt")
    cresults_output_path = Path("/home/v-asonwane/cai/data/traj_type/full_traj_type_results.json")
    bug_type_counts_output_path = Path("/home/v-asonwane/cai/data/traj_type/traj_type_counts.json")

    pprint(dataset_paths)

    all_sums = {}
    for d_path in dataset_paths:
        data = [json.loads(l) for l in open(d_path, "r").read().splitlines()]
        d_sums = analyse(data, sample_n=16, k=16)
        all_sums[d_path] = d_sums

    print(f"Writing summaries to {cumsum_output_path}")
    json.dump(all_sums, open(cumsum_output_path, "w"), indent=4)


    all_sums = json.load(open(cumsum_output_path, "r"))

    type_sums = []
    for k, v in all_sums.items():
        type_sums.append(v[-1][0])

    tss = ""
    for idx, ts in enumerate(type_sums):
        tss += f"<idx_{idx}>\n{ts}\n</idx_{idx}>\n\n"

    prompt = f"""Below is a list of trajectory summaries generalised from descriptions of bug-fixing trajectories.
Your task is to produce a list of 10 types from this list.

{tss}

For each type, specify the name and description. 
Include enough information so that given a new trajectory it can be categorised.
"""

    llm_response = completion(
        MODEL,
        messages=[{"role": "user", "content": prompt}],
        azure_ad_token_provider=AZURE_AD_TOKEN_PROVIDER,
    )
    summary = llm_response["choices"][0]["message"]["content"]        

    print(f"Writing guidelines to {summary_output_path}")
    open(summary_output_path, "w").write(summary)

    # guide = open(summary_output_path, "r").read()
    # full_bug_type_results = {}
    # bug_counts = {}
    # for d_path in dataset_paths:
    #     data = json.load(open(d_path, "r"))
    #     cresults = categorise(guide, data)
    #     full_bug_type_results[d_path] = cresults

    #     ccounts = defaultdict(lambda: 0)
    #     for bug, res in cresults:
    #         ccounts[res] += 1
    #     bug_counts[d_path] = ccounts

    # print(bug_counts)

    # print(f"Saving full bug type counts to {cresults_output_path}")
    # json.dump(full_bug_type_results, open(cresults_output_path, "w"), indent=2)

    # print(f"Saving bug counts to {bug_type_counts_output_path}")
    # json.dump(bug_counts, open(bug_type_counts_output_path, "w"), indent=2)

    

if __name__ == "__main__":
    main()
