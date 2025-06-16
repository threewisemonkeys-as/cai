import json
from pathlib import Path
from collections import defaultdict

def analyse_matching(
    correct: Path | str,
    log_dir: Path | str,    
):
    data = json.load(open(correct, "r"))
    pos_file_names = data['matching_files']
    pos_bug_types = [
        n.split('.')[-1].split('__')[0]
        for n in pos_file_names
    ]
    pos_bug_types = [t for t in pos_bug_types if "pr_" not in t]

    bug_type_counts = defaultdict(lambda: 0)
    for bt in pos_bug_types: bug_type_counts[bt] += 1
    print(bug_type_counts)

    all_bug_types = []
    all_bug_types_count = defaultdict(lambda: 0)
    for log_file in Path(log_dir).rglob("*.log"):
        bug_type = log_file.name.split('.')[-2].split('__')[0]
        if "pr_" in bug_type:
            continue
        all_bug_types.append(bug_type)
        all_bug_types_count[bug_type] += 1
    
    ratios = {}
    for bt in all_bug_types_count:
        ratios[bt] = bug_type_counts[bt] / all_bug_types_count[bt]
    for bt, btr in sorted(ratios.items(), key=lambda r: r[-1], reverse=True):
        print(f"{bt}\t\t\t{bug_type_counts[bt]}\t{all_bug_types_count[bt]}\t{btr:.4f}")


def analyse(
    log_dir: Path | str,
):
    status = {}
    for log_file in Path(log_dir).rglob("*debug_gym.jsonl"):
        data = json.load(open(log_file, "r"))
        status[data['problem']] = {
            'success': data['success'],
            'pdb_count': sum(d['action']['name'] == 'pdb' if d['action'] is not None else 0 for d in data['log']),
            'traj_len': len(data['log']),
        }
    
    correct_count = sum(s['success'] == True for s in status.values())
    print(f"Correct: {correct_count} / {len(status)} = {correct_count / len(status):.4f}")

    print(f"Avg traj length: {sum(d['traj_len'] for d in status.values()) / len(status):.2f}")
    print(f"Avg pdb usage: {sum(d['pdb_count'] for d in status.values()) / len(status):.2f}")


if __name__ == '__main__':
    import fire
    fire.Fire(dict(
        matching=analyse_matching,
        analyse=analyse,
    ))