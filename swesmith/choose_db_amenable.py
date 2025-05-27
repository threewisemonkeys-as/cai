import litellm
from tqdm import tqdm
from athils import JsonLinesFile

from dotenv import load_dotenv

load_dotenv()


K = None
USE_FIX = False
MODEL = "azure/o3_2025-04-16"

SELECTION_PROMPT = r"""You are given information about a bug report filed in a github repository.
Your task is to decide whether using the python debugger (pdb) is necesary in order to diagnose and resolve the issue.

Problem statement:
<problem_statement>
{problem_statement}
</problem_statement>

Correct fix:
<fix>
{fix}
</fix>

Given this information about this bug, would it have been posibble to diagnose and resolve purely by executing tests and reproduction scripts or is it necessary to use the python debugger (pdb) to gain further visbility into the runtime behavior of the codebase.
At the end of your answer, include answer=yes if you think pdb is necessary or answer=no if you think pdb is not necessary to diagnose and resolve this task.
"""

SELECTION_PROMPT_WITHOUT_FIX = r"""You are given information about a bug report filed in a github repository.
Your task is to decide whether using the python debugger (pdb) is necesary in order to diagnose and resolve the issue.

Problem statement:
<problem_statement>
{problem_statement}
</problem_statement>


Given this information about this bug, would it have been posibble to diagnose and resolve purely by executing tests and reproduction scripts or is it necessary to use the python debugger (pdb) to gain further visbility into the runtime behavior of the codebase.
At the end of your answer, include answer=yes if you think pdb is necessary or answer=no if you think pdb is not necessary to diagnose and resolve this task.
"""


input_path = "data/swesmith/subset.jsonl"
output_path = "data/swesmith/db_amenable_wo_fix_o3.jsonl"

data = JsonLinesFile.read_from(input_path)

for idx, d in tqdm(enumerate(data), desc="Processing instances", total=len(data) if K is None else K):

    if K is not None and idx >= K:
        break


    if USE_FIX:
        prompt = SELECTION_PROMPT.format(
            problem_statement=d['problem_statement'],
            fix=d['patch'],
        )
    else:
        prompt = SELECTION_PROMPT_WITHOUT_FIX.format(
            problem_statement=d['problem_statement'],
        )

    response = litellm.completion(
        model = MODEL,
        messages = [{"role": "user", "content": prompt}],
    )

    response_text = response.choices[0].message.content

    response_clean = ''.join(response_text.lower().split())
    answer = None
    if "answer=yes" in response_clean:
        answer = True
    elif "answer=no" in response_clean:
        answer = False

    result = {
        "instance_id": d['instance_id'],
        "problem_statement": d['problem_statement'],
        "selection_response": response_text,
        "selection_answer": answer,
    }

    JsonLinesFile.add_to(output_path, result)