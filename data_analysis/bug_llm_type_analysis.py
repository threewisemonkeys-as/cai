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
                summary_prompt = f"""Generalise possible bug types from the provided bug summaries.

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
    for bug in random.sample(data, sample_n):   
        if "patch" in bug:
            patch = bug['patch']
        elif "parsed_commit_content" in bug:
            commit = ParsedCommit(**json.loads(bug['parsed_commit_content']))
            patch = commit.get_patch()
        else:        
            raise RuntimeError(f"Count not find relevent key for patch in {bug.keys()}")
        ps = bug['problem_statement']
        prompt = f"""Your task is to summarise in brief the bug presented here.

<problem_description>
{ps}
</problem_description>

<patch>
{patch}
</patch>

Considering the given problem description and patch, summarise the bug in brief, including characteristics of the type of bug."""
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
    for bug in tqdm(samples):
        if "patch" in bug:
            patch = bug['patch']
        elif "parsed_commit_content" in bug:
            commit = ParsedCommit(**json.loads(bug['parsed_commit_content']))
            patch = commit.get_patch()
        else:        
            raise RuntimeError(f"Count not find relevent key for patch in {bug.keys()}")
        ps = bug['problem_statement']
        prompt = f"""Your task is to categorise a provided bug into a set of given bug types.

Here are the guidelines on the bug types - 
{guide}

Here is the bug the needs to be categoried -

<problem_description>
{ps}
</problem_description>

<patch>
{patch}
</patch>

Your response should be in xml format:
<reasoning>
Thinking about which categories that the given bug falls into.
</reasoning>
<category>
Alphabet code of category that bug falls into.
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
        "/home/v-asonwane/cai/data/rl_tasks/swebv.json"
        # "/home/v-asonwane/cai/data/rl_tasks/buggen_train.json",
        # "/home/v-asonwane/cai/data/rl_tasks/swesmith_train.json",
        # "/home/v-asonwane/cai/data/rl_tasks/buggen_train.json",
        # "/home/v-asonwane/cai/data/rl_tasks/featadd_train.json",
    ]

    cumsum_output_path = Path("/home/v-asonwane/cai/data/bug_type/llm_bugtype.json")
    summary_output_path = Path("/home/v-asonwane/cai/data/bug_type/bug_types.txt")
    cresults_output_path = Path("/home/v-asonwane/cai/data/bug_type/full_bug_type_results.json")
    bug_type_counts_output_path = Path("/home/v-asonwane/cai/data/bug_type/bug_type_counts.json")

    pprint(dataset_paths)

    # all_sums = {}
    # for d_path in dataset_paths:
    #     data = json.load(open(d_path, "r"))
    #     d_sums = analyse(data, sample_n=16, k=16)
    #     all_sums[d_path] = d_sums

    # print(f"Writing summaries to {cumsum_output_path}")
    # json.dump(all_sums, open(cumsum_output_path, "w"), indent=4)


#     all_sums = json.load(open(cumsum_output_path, "r"))

#     type_sums = []
#     for k, v in all_sums.items():
#         type_sums.append(v[-1][0])

#     tss = ""
#     for idx, ts in enumerate(type_sums):
#         tss += f"<idx_{idx}>\n{ts}\n</idx_{idx}>\n\n"

#     prompt = f"""Below is a list of bug types generalised from some bug descriptions.
# Your task is to produce a list of 10 bug types from this list.

# {tss}

# For each bug type, specify the name and description of the bug. 
# Include enough information so that given a new bug it can be categorised.
# """

#     llm_response = completion(
#         "openai/gpt-5",
#         messages=[{"role": "user", "content": prompt}],
#         azure_ad_token_provider=AZURE_AD_TOKEN_PROVIDER,
#     )
#     summary = llm_response["choices"][0]["message"]["content"]        

#     print(f"Writing guidelines to {summary_output_path}")
#     open(summary_output_path, "w").write(summary)

    guide = open(summary_output_path, "r").read()
    full_bug_type_results = {}
    bug_counts = {}
    for d_path in dataset_paths:
        data = json.load(open(d_path, "r"))
        cresults = categorise(guide, data)
        full_bug_type_results[d_path] = cresults

        ccounts = defaultdict(lambda: 0)
        for bug, res in cresults:
            ccounts[res] += 1
        bug_counts[d_path] = ccounts

    print(bug_counts)

    print(f"Saving full bug type counts to {cresults_output_path}")
    json.dump(full_bug_type_results, open(cresults_output_path, "w"), indent=2)

    print(f"Saving bug counts to {bug_type_counts_output_path}")
    json.dump(bug_counts, open(bug_type_counts_output_path, "w"), indent=2)

    

if __name__ == "__main__":
    main()
