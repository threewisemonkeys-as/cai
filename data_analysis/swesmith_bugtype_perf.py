#%%

import json
from pathlib import Path

from rich.table import Table
from rich.console import Console

#%%

def plot_success_rate_vs_patch_size(patch_size_data, title_suffix="", filename_suffix=""):
    """Plot success rate vs patch size using 5 equal-sized buckets"""
    import matplotlib.pyplot as plt
    
    if not patch_size_data:
        print(f"No patch size data available for {title_suffix}")
        return
    
    # Sort patch size data by patch size
    sorted_data = sorted(patch_size_data, key=lambda x: x['patch_size'])
    
    # Use percentile-based bucketing to handle skewed data better
    import numpy as np
    patch_sizes = [item['patch_size'] for item in patch_size_data]
    
    # Use percentiles that give more granular view of the distribution
    percentiles = [0, 25, 50, 75, 95, 100]  # 5 buckets with finer granularity at high end
    bucket_thresholds = np.percentile(patch_sizes, percentiles)
    
    # Ensure unique thresholds (in case of many identical values)
    unique_thresholds = []
    for threshold in bucket_thresholds:
        if not unique_thresholds or threshold > unique_thresholds[-1]:
            unique_thresholds.append(threshold)
    
    # If we don't have enough unique thresholds, fall back to fixed ranges
    if len(unique_thresholds) < 3:
        max_size = max(patch_sizes)
        unique_thresholds = [0, max_size//4, max_size//2, max_size*3//4, max_size]
    
    buckets = []
    for i in range(len(unique_thresholds) - 1):
        min_threshold = unique_thresholds[i]
        max_threshold = unique_thresholds[i + 1]
        
        # Include items in this range
        if i == len(unique_thresholds) - 2:  # Last bucket includes max value
            bucket_items = [item for item in patch_size_data 
                          if min_threshold <= item['patch_size'] <= max_threshold]
        else:
            bucket_items = [item for item in patch_size_data 
                          if min_threshold <= item['patch_size'] < max_threshold]
        
        if bucket_items:  # Only create bucket if it has items
            success_count = sum(1 for item in bucket_items if item['success'])
            total_count = len(bucket_items)
            success_rate = success_count / total_count if total_count > 0 else 0
            
            buckets.append({
                'success_rate': success_rate,
                'total': total_count,
                'min_size': int(min_threshold),
                'max_size': int(max_threshold),
                'label': f"{int(min_threshold)}-{int(max_threshold)}"
            })

    success_rates = [bucket['success_rate'] for bucket in buckets]
    bucket_labels = [bucket['label'] for bucket in buckets]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(bucket_labels)), success_rates, color='steelblue', alpha=0.7)

    plt.xlabel('Patch Size Range (lines changed)')
    plt.ylabel('Average Success Rate')
    plt.title(f'Average Success Rate vs {title_suffix} Patch Size (Percentile-Based Buckets)')
    plt.xticks(range(len(bucket_labels)), bucket_labels, rotation=45)
    plt.ylim(0, 1)

    for i, (bar, bucket) in enumerate(zip(bars, buckets)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{bucket["success_rate"]:.2f}\n(n={bucket["total"]})', 
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{filename_suffix}_patch_size_success_rate.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_success_rate_vs_repo_size(repo_size_data, title_suffix="Repo", filename_suffix="repo"):
    """Plot success rate vs repo size using percentile-based buckets"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not repo_size_data:
        print(f"No repo size data available")
        return
    
    # Use percentile-based bucketing to handle skewed data better
    repo_sizes = [item['repo_size'] for item in repo_size_data]
    
    # Use percentiles that give more granular view of the distribution
    percentiles = [0, 25, 50, 75, 95, 100]  # 5 buckets with finer granularity at high end
    bucket_thresholds = np.percentile(repo_sizes, percentiles)
    
    # Ensure unique thresholds (in case of many identical values)
    unique_thresholds = []
    for threshold in bucket_thresholds:
        if not unique_thresholds or threshold > unique_thresholds[-1]:
            unique_thresholds.append(threshold)
    
    # If we don't have enough unique thresholds, fall back to fixed ranges
    if len(unique_thresholds) < 3:
        max_size = max(repo_sizes)
        unique_thresholds = [0, max_size//4, max_size//2, max_size*3//4, max_size]
    
    buckets = []
    for i in range(len(unique_thresholds) - 1):
        min_threshold = unique_thresholds[i]
        max_threshold = unique_thresholds[i + 1]
        
        # Include items in this range
        if i == len(unique_thresholds) - 2:  # Last bucket includes max value
            bucket_items = [item for item in repo_size_data 
                          if min_threshold <= item['repo_size'] <= max_threshold]
        else:
            bucket_items = [item for item in repo_size_data 
                          if min_threshold <= item['repo_size'] < max_threshold]
        
        if bucket_items:  # Only create bucket if it has items
            success_count = sum(1 for item in bucket_items if item['success'])
            total_count = len(bucket_items)
            success_rate = success_count / total_count if total_count > 0 else 0
            
            # Format repo size labels with appropriate units
            min_size_label = format_repo_size(int(min_threshold))
            max_size_label = format_repo_size(int(max_threshold))
            
            buckets.append({
                'success_rate': success_rate,
                'total': total_count,
                'min_size': int(min_threshold),
                'max_size': int(max_threshold),
                'label': f"{min_size_label}-{max_size_label}"
            })

    success_rates = [bucket['success_rate'] for bucket in buckets]
    bucket_labels = [bucket['label'] for bucket in buckets]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(bucket_labels)), success_rates, color='darkgreen', alpha=0.7)

    plt.xlabel('Repo Size Range')
    plt.ylabel('Average Success Rate')
    plt.title(f'Average Success Rate vs {title_suffix} Size (Percentile-Based Buckets)')
    plt.xticks(range(len(bucket_labels)), bucket_labels, rotation=45)
    plt.ylim(0, 1)

    for i, (bar, bucket) in enumerate(zip(bars, buckets)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{bucket["success_rate"]:.2f}\n(n={bucket["total"]})', 
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(f"{filename_suffix}_size_success_rate.png", dpi=300, bbox_inches='tight')
    plt.show()


def format_repo_size(size):
    """Format repo size with appropriate units (KB, MB, GB)"""
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size//1024}KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size//(1024*1024)}MB"
    else:
        return f"{size//(1024*1024*1024)}GB"


def analyse(
    results_dir: Path | str,
    data_path: Path | str | None = None,
    repo_sizes: Path | str | None = None,
):

    if data_path is not None:
        data = json.load(open(data_path, "r"))
        data = {i['instance_id']: i for i in data}
    else:
        data = None
    
    if repo_sizes is not None:
        repo_size = [
            json.loads(line)
            for line in Path(repo_sizes).read_text().splitlines()
        ]
        repo_size = {'.'.join(i[0].split('.')[2:]): int(i[1]) for i in repo_size}

    results_dir = Path(results_dir)

    skipped = 0
    bug_type_results = {}
    result_patch_size_map = {True: (0, 0), False: (0, 0)}
    patch_size_data = []

    result_gold_patch_size_map = {True: (0, 0), False: (0, 0)}
    gold_patch_size_data = []

    result_repo_size_map = {True: (0, 0), False: (0, 0)}
    repo_size_data = []

    for subdir in results_dir.iterdir():
        result_name = subdir.name
        name_parts = result_name.split('.')
        if len(name_parts) not in [3, 4]:
            print(f"Skipping {result_name} due to unexpected format")
            skipped += 1
            continue


        if not (subdir / "debug_gym.jsonl").exists():
            print(f"Skipping {result_name} as debug_gym.jsonl does not exist")
            skipped += 1
            continue

        dbgym_result = json.load((subdir / "debug_gym.jsonl").open("r"))
        dbgym_patch = (subdir / "debug_gym.patch").read_text()

        success = dbgym_result.get('success', False)

        if len(name_parts) == 4:
            repo, commit, bug_info, seed = name_parts
        else:
            repo, commit, bug_info = name_parts
            seed = None
        bug_info_split = bug_info.split('__')

        bug_type = bug_info_split[0]

        if '_' in bug_type:
            bug_type_short = bug_type.split('_')[0] 
        else:
            bug_type_short = bug_type
        
        if bug_type_short not in bug_type_results:
            bug_type_results[bug_type_short] = (0, 0)
        
        success_count, total_count = bug_type_results[bug_type_short]
        success_count += 1 if success else 0
        total_count += 1
        bug_type_results[bug_type_short] = (success_count, total_count)


        if (subdir / "debug_gym.patch").exists():
            dbgym_patch = (subdir / "debug_gym.patch").read_text()
            if dbgym_patch.strip() != "":
                patch_size = sum([l.startswith('+') or l.startswith('-') for l in dbgym_patch.splitlines()])
                
                result_patch_size_total, result_count = result_patch_size_map[success]
                result_count += 1
                result_patch_size_total += patch_size
                result_patch_size_map[success] = (result_patch_size_total, result_count)
                
                patch_size_data.append({'patch_size': patch_size, 'success': success})

        if data is not None and result_name in data:
            gold_patch = data[result_name].get('patch', '')
            if gold_patch.strip() != "":
                gold_patch_size = sum([l.startswith('+') or l.startswith('-') for l in gold_patch.splitlines()])
                
                result_gold_patch_size_total, result_gold_count = result_gold_patch_size_map[success]
                result_gold_count += 1
                result_gold_patch_size_total += gold_patch_size
                result_gold_patch_size_map[success] = (result_gold_patch_size_total, result_gold_count)
                
                gold_patch_size_data.append({'patch_size': gold_patch_size, 'success': success})

        # Collect repo size data if available
        if repo_sizes is not None:
            repo_commit_key = f"{repo}.{commit}"
            if repo_commit_key in repo_size:
                repo_size_value = repo_size[repo_commit_key]
                
                result_repo_size_total, result_repo_count = result_repo_size_map[success]
                result_repo_count += 1
                result_repo_size_total += repo_size_value
                result_repo_size_map[success] = (result_repo_size_total, result_repo_count)
                
                repo_size_data.append({'repo_size': repo_size_value, 'success': success})



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

    # Gold patch size table
    if gold_patch_size_data:
        table = Table(title="Gold Patch Size by Result", show_lines=False)
        table.add_column("Result", justify="right", style="cyan")
        table.add_column("Total", justify="right", style="magenta")
        table.add_column("Avg. Gold Patch Size", justify="right", style="yellow")

        for result, (size, tot) in sorted(
            result_gold_patch_size_map.items(),
            key=lambda x: x[1][0] / x[1][1] if x[1][1] else 0,
            reverse=True,
        ):
            ratio = size / tot if tot else 0
            table.add_row(str(result), str(tot), f"{ratio:.2f}")

        Console().print(table)

    # Repo size table
    if repo_size_data:
        table = Table(title="Repo Size by Result", show_lines=False)
        table.add_column("Result", justify="right", style="cyan")
        table.add_column("Total", justify="right", style="magenta")
        table.add_column("Avg. Repo Size", justify="right", style="yellow")

        for result, (size, tot) in sorted(
            result_repo_size_map.items(),
            key=lambda x: x[1][0] / x[1][1] if x[1][1] else 0,
            reverse=True,
        ):
            ratio = size / tot if tot else 0
            table.add_row(str(result), str(tot), f"{ratio:.0f}")

        Console().print(table)

    #%%
    # Plot actual patch sizes
    plot_success_rate_vs_patch_size(patch_size_data, "Actual", "actual")
    
    # Plot gold patch sizes if available
    if gold_patch_size_data:
        plot_success_rate_vs_patch_size(gold_patch_size_data, "Gold", "gold")
    
    # Plot repo sizes if available
    if repo_size_data:
        plot_success_rate_vs_repo_size(repo_size_data, "Repo", "repo")




if __name__ == "__main__":
    import fire
    fire.Fire(analyse)