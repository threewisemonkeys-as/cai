from pathlib import Path
import json
from collections import defaultdict
import random

def collect(
    input_data: Path | str,
):
    input_data = Path(input_data)
    data = [json.loads(d) for d in input_data.read_text().splitlines()]
    
    per_problem = defaultdict(list)
    for d in data:
        per_problem[d['problem_statement']].append(
            {'attempt': d['attempt'], 'reward': d['reward']}
        )
    
    perf = sum([
        1.0 if sum(p['reward'] for p in pdata) > 0 else 0.0
        for pdata in per_problem.values()
    ]) / len(per_problem)

def collect_at_k(
    input_data: Path | str,
    k: int = 10,
):
    input_data = Path(input_data)
    data = [json.loads(d) for d in input_data.read_text().splitlines()]
    
    per_problem = defaultdict(list)
    for d in data:
        per_problem[d['problem_statement']].append(
            {'attempt': d['attempt'], 'reward': d['reward']}
        )

    print(f"Num problems: {len(per_problem)}")
    print(f"Av k per problem: {sum(len(pdata) for pdata in per_problem.values()) / len(per_problem):.2f}")
    
    per_k = defaultdict(list)
    for i in range(1, k + 1):
        for pdata in per_problem.values():
            selected = random.sample(pdata, k=min(i, len(pdata)))
            per_k[i].append(
                1.0 if sum(d['reward'] for d in selected) > 0 else 0.0
            )
    
    per_k_perf = {
        i: sum(i_data) / len(i_data)
        for i, i_data in per_k.items()
    }

    for i in range(1, k + 1):
        print(f"{i}\t{per_k_perf[i]:.2f}")



if __name__ == '__main__':
    import fire
    fire.Fire({
        "collect": collect,
        "collect_at_k": collect_at_k,
    })
