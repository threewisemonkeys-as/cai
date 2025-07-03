import json
import math
from pathlib import Path
from collections import defaultdict

from athils import JsonLinesFile

from utils import snip_trace

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



def analyse_detail(
    log_dir: Path | str,
):
    status = {}
    for log_file in Path(log_dir).rglob("*debug_gym.jsonl"):
        data = json.load(open(log_file, "r"))
        pdb_count = sum(d['action']['name'] == 'pdb' if d['action'] is not None else 0 for d in data['log'])
        status[data['problem']] = {
            'success': data['success'],
            'pdb_count': pdb_count,
            'traj_len': len(data['log']),
        }
        
        if pdb_count > 0 and data['success']:
            print(log_file, pdb_count)



def analyse_with_trace(
    log_dir: Path | str,
    traces: Path | str,
):
    avg = lambda l: sum(l) / len(l) if len(l) > 0 else None
    std_p = lambda l: math.sqrt(sum((x - avg(l))**2 for x in l) / len(l)) if l else None

    status = {}
    for log_file in Path(log_dir).rglob("*debug_gym.jsonl"):
        data = json.load(open(log_file, "r"))
        status[data['problem']] = {
            'success': data['success'],
            'traj_len': len(data['log']),
        }
            
    
    trace_data = JsonLinesFile.read_from(traces)
    trace_map = {d[0]['instance_id']: d for d in trace_data}

    num_tests = []
    passed, failed = [], []

    successfull_max_depths, failed_max_depths = [], []
    successfull_lens, failed_lens = [], []
    successfull_unique, failed_unique = [], []
    for problem, data in status.items():
        if problem not in trace_map:
            print(f"Could not find {problem} in traces file {traces}")
            continue

        test_depths, test_lens, test_uniques = [], [], []
        traces_data = trace_map[problem][1]
        
        _num_tests = len(traces_data)
        if _num_tests > 1:
            continue

        for test, test_trace in traces_data:
            try:
                curr_trace = snip_trace(test_trace['trace_data'], test)
            except:
                print(f"Could not find {test} in trace data")
                continue
                curr_trace = test_trace['trace_data']
            
            test_depths.append(max(e['depth'] for e in curr_trace))
            test_lens.append(len(curr_trace))
            test_uniques.append(len(set([(e['location'], e['name']) for e in curr_trace])))

        num_tests.append(_num_tests)
        md = max(test_depths)
        al = avg(test_lens)
        au = avg(test_uniques)
        if data['success']:
            passed.append(problem)
            successfull_max_depths.append(md)
            successfull_lens.append(al)
            successfull_unique.append(au)
        else:
            failed.append(problem)
            failed_max_depths.append(md)
            failed_lens.append(al)
            failed_unique.append(au)
    
    def print_stats(succ_data: list, fail_data: list):
        print(f"{avg(succ_data)=:.2f} {avg(fail_data)=:.2f}")
        print(f"{std_p(succ_data)=:.2f} {std_p(fail_data)=:.2f}")
    
    print(f"Max depth stats - ")
    print(f"{successfull_max_depths=}")
    print(f"{failed_max_depths=}")
    print_stats(successfull_max_depths, failed_max_depths)


    print(f"\n\nLength stats - ")
    print(f"{successfull_lens=}")
    print(f"{failed_lens=}")
    print_stats(successfull_lens, failed_lens)

    print(f"\n\nUnique stats - ") 
    print(f"{successfull_unique=}")
    print(f"{failed_unique=}")
    print_stats(successfull_unique, failed_unique)

    print(f"\n\nNumber of tests: {num_tests} ")

    print(f'\n\n{passed=}')
    print(f'\n\n{failed=}')

if __name__ == '__main__':
    import fire
    fire.Fire(dict(
        matching=analyse_matching,
        analyse=analyse,
        detail=analyse_detail,
        with_trace=analyse_with_trace,
    ))