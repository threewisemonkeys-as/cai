#%%

import json
from pathlib import Path

from rich.table import Table
from rich.console import Console

#%%



results_dir = Path("/home/t-atsonwane/work/debug-gym/exps/swesmith_bugs/jul14/debug_o4mini_0")

skipped = 0
bug_type_results = {}
result_patch_size_map = {True: (0, 0), False: (0, 0)}

for subdir in results_dir.iterdir():
    result_name = subdir.name
    name_parts = result_name.split('.')
    if len(name_parts) != 3:
        print(f"Skipping {result_name} due to unexpected format")
        skipped += 1
        continue


    if not (subdir / "debug_gym.jsonl").exists():
        print(f"Skipping {result_name} as debug_gym.jsonl does not exist")
        skipped += 1
        continue

    if not (subdir / "debug_gym.patch").exists():
        print(f"Skipping {result_name} as debug_gym.patch does not exist")
        skipped += 1
        continue

    dbgym_result = json.load((subdir / "debug_gym.jsonl").open("r"))
    dbgym_patch = (subdir / "debug_gym.patch").read_text()
    patch_size = sum([l.startswith('+') or l.startswith('-') for l in dbgym_patch.splitlines()])

    success = dbgym_result.get('success', False)

    repo, commit, bug_info = name_parts
    bug_info_split = bug_info.split('__')

    if len(bug_info_split) != 2:
        print(f"Skipping {result_name} due to unexpected bug info format")
        continue

    bug_type = bug_info_split[0]

    bug_type_short = bug_type.split('_')[0] 
    if bug_type_short not in bug_type_results:
        bug_type_results[bug_type_short] = (0, 0)
    
    success_count, total_count = bug_type_results[bug_type_short]
    success_count += 1 if success else 0
    total_count += 1
    bug_type_results[bug_type_short] = (success_count, total_count)

    result_patch_size_total, result_count = result_patch_size_map[success]
    result_count += 1
    result_patch_size_total += patch_size
    result_patch_size_map[success] = (result_patch_size_total, result_count)

print(f"Skipped = {skipped}")

table = Table(title="Debug‑Gym Results by Bug Type", show_lines=False)
table.add_column("Bug Type", style="cyan", no_wrap=True)
table.add_column("Success", justify="right", style="green")
table.add_column("Total", justify="right", style="magenta")
table.add_column("Ratio", justify="right", style="yellow")

for bug_type, (succ, tot) in sorted(
    bug_type_results.items(),
    key=lambda x: x[1][0] / x[1][1] if x[1][1] else 0,
    reverse=True,
):
    ratio = succ / tot if tot else 0
    table.add_row(bug_type, str(succ), str(tot), f"{ratio:.4f}")

Console().print(table)



table = Table(title="Patch Size by Result", show_lines=False)
table.add_column("Result", justify="right", style="cyan")
table.add_column("Total", justify="right", style="magenta")
table.add_column("Avg. Patch Size", justify="right", style="yellow")

for result, (size, tot) in sorted(
    result_patch_size_map.items(),
    key=lambda x: x[1][0] / x[1][1] if x[1][1] else 0,
    reverse=True,
):
    ratio = size / tot if tot else 0
    table.add_row(str(result), str(tot), f"{ratio:.2f}")


Console().print(table)
