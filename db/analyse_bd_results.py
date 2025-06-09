import json
from pathlib import Path
from collections import defaultdict

def analyse(
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



if __name__ == '__main__':
    import fire
    fire.Fire(analyse)