#%%

import os
from datasets import load_dataset
from athils import JsonLinesFile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


analysis_data = JsonLinesFile.read_from("/home/t-atsonwane/work/cai/data/swebench/verified_traces_analysis.jsonl")
analysis_data_map = {d['instance_id']: d for d in analysis_data}

swebv = load_dataset("SWE-bench/SWE-bench_Verified")['test']

swebv_disc = load_dataset("jatinganhotra/SWE-bench_Verified-discriminative")
swebv_disc_map = {
    iid: k
    for k in swebv_disc.keys()
    for iid in swebv_disc[k]['instance_id']
}


#%%

import ast
import re
import textwrap
from pathlib import PurePosixPath
from typing import Dict, List, Set

import requests


RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def _hunks_by_file(patch: str) -> Dict[str, Set[int]]:
    """
    Parse a unified diff and return, per file, the set of *new*‐file
    line numbers that were added/changed.
    """
    file_changes: Dict[str, Set[int]] = {}
    current_file = None
    new_ln = 0

    hunk_re = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")  # +start,len

    for line in patch.splitlines():
        if line.startswith("diff --git"):
            # e.g. diff --git a/foo.py b/foo.py
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[3][2:]  # strip leading "b/"
                file_changes[current_file] = set()
            continue

        if current_file is None:
            continue

        if line.startswith("@@"):
            m = hunk_re.match(line)
            if not m:
                continue
            new_ln = int(m.group(1))
            continue

        # Skip file header lines such as "---", "+++"
        if line.startswith(("---", "+++", "diff --git")):
            continue

        # Within a hunk
        if line.startswith("+") and not line.startswith("+++"):
            file_changes[current_file].add(new_ln)
            new_ln += 1
        elif not line.startswith("-"):
            # context line
            new_ln += 1
        # for deletions ('-') we do NOT advance the new-file line counter

    # Remove entries where we never recorded any +lines
    return {f: lns for f, lns in file_changes.items() if lns}


def _get_py_file(repo: str, commit: str, path: str, token: str | None = None) -> str:
    url = RAW_URL.format(repo=repo, commit=commit, path=path)
    headers = {"Authorization": f"token {token}"} if token else {}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.text


def _func_spans(source: str) -> List[tuple[int, int]]:
    """
    Return [(start_lineno, end_lineno)] for every FunctionDef / AsyncFunctionDef
    in *source*, using end_lineno (Py≥3.8) or a manual fallback.
    """
    tree = ast.parse(source)
    spans: List[tuple[int, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", None)
            if end is None:  # <3.8: find via body of function
                if node.body:
                    end = max(getattr(n, "end_lineno", n.lineno) for n in node.body)
                else:
                    end = start
            spans.append((start, end))
    return spans


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_modified_func_sizes(patch: str, repo_name: str, base_commit: str) -> List[int]:
    """
    Given a unified diff *patch*, return the source-line counts of every Python
    function that the patch modifies.  Duplicate functions (touched in multiple
    hunks) appear only once in the result.
    """
    modified_sizes: List[int] = []

    for path, touched_lines in _hunks_by_file(patch).items():
        # We only care about Python files
        if PurePosixPath(path).suffix != ".py":
            continue

        try:
            source = _get_py_file(repo_name, base_commit, path, token=GITHUB_TOKEN)
        except requests.HTTPError:
            # File might be deleted/renamed; ignore it
            continue

        spans = _func_spans(source)

        seen = set()
        for start, end in spans:
            # Any overlap between touched lines and this function span?
            if any(start <= ln <= end for ln in touched_lines):
                if (start, end) not in seen:
                    modified_sizes.append(end - start + 1)
                    seen.add((start, end))

    return modified_sizes



#%%

data = []


for instance in swebv:
    instance_id = instance['instance_id']

    if instance_id not in analysis_data_map:
        print(f"Could not find {instance_id}")
        continue

    if instance_id not in swebv_disc_map:
        print(f"Could not find {instance_id}")
        continue

    instance_analysis = analysis_data_map[instance_id]

    modified_func_sizes = get_modified_func_sizes(
        patch=instance['patch'],
        repo_name=instance['repo'],
        base_commit=instance['base_commit'],
    )
    avg_modified_func_size = sum(modified_func_sizes) / len(modified_func_sizes) if modified_func_sizes is not None and len(modified_func_sizes) > 0 else None
    

    idata = {
        "instance_id": instance_id,
        "human_difficulty": instance['difficulty'],
        "agent_difficulty": swebv_disc_map[instance_id], 
        "modified_func_size": avg_modified_func_size,
        **instance_analysis
    }

    data.append(idata)
    




#%%

df = pd.DataFrame(data)

#%%
# Analysis of max_depth and unique_calls by difficulty

def analyze_instance_metrics(df, name, metrics: list[str]):
    """
    Analyze max_depth and unique_calls metrics grouped by difficulty
    """
    # Group by and calculate statistics
    stats = df.groupby(name).agg({
        metric_name: ['mean', 'std', 'count']
        for metric_name in metrics
    }).round(3)
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns]
    
    print(f"Statistics by {name}:")
    print("=" * 50)
    print(stats)
    print()
    
    # Create detailed summary
    summary_data = []
    for cat in df[name].unique():
        subset = df[df[name] == cat]
        means_dict = {
            f"{metric_name}_mean": subset[metric_name].mean()
            for metric_name in metrics
        }
        std_dict = {
            f"{metric_name}_std": subset[metric_name].std()
            for metric_name in metrics
        }
        summary_data.append({
            name: cat,
            'count': len(subset),
            **means_dict,
            **std_dict,
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("Detailed Summary:")
    print(summary_df.to_string(index=False))
    print()
    
    return stats, summary_df

def create_plots(df):
    """
    Create visualizations for max_depth and unique_calls by difficulty
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Instance Analysis: Max Length and Unique Calls by Difficulty', fontsize=16, fontweight='bold')
    
    # Plot 1: Box plots for max_depth
    sns.boxplot(data=df, x='difficulty', y='max_depth', ax=axes[0, 0])
    axes[0, 0].set_title('Max Length Distribution by Difficulty')
    axes[0, 0].set_xlabel('Difficulty')
    axes[0, 0].set_ylabel('Max Length')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Box plots for unique_calls
    sns.boxplot(data=df, x='difficulty', y='unique_calls', ax=axes[0, 1])
    axes[0, 1].set_title('Unique Calls Distribution by Difficulty')
    axes[0, 1].set_xlabel('Difficulty')
    axes[0, 1].set_ylabel('Unique Calls')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Violin plots for max_depth
    sns.violinplot(data=df, x='difficulty', y='max_depth', ax=axes[0, 2])
    axes[0, 2].set_title('Max Length Density by Difficulty')
    axes[0, 2].set_xlabel('Difficulty')
    axes[0, 2].set_ylabel('Max Length')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Plot 4: Bar plot of means with error bars for max_depth
    stats_max_depth = df.groupby('difficulty')['max_depth'].agg(['mean', 'std']).reset_index()
    axes[1, 0].bar(stats_max_depth['difficulty'], stats_max_depth['mean'], 
                   yerr=stats_max_depth['std'], capsize=5, alpha=0.7)
    axes[1, 0].set_title('Max Length: Mean ± Std Dev by Difficulty')
    axes[1, 0].set_xlabel('Difficulty')
    axes[1, 0].set_ylabel('Max Length')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Bar plot of means with error bars for unique_calls
    stats_unique_calls = df.groupby('difficulty')['unique_calls'].agg(['mean', 'std']).reset_index()
    axes[1, 1].bar(stats_unique_calls['difficulty'], stats_unique_calls['mean'], 
                   yerr=stats_unique_calls['std'], capsize=5, alpha=0.7)
    axes[1, 1].set_title('Unique Calls: Mean ± Std Dev by Difficulty')
    axes[1, 1].set_xlabel('Difficulty')
    axes[1, 1].set_ylabel('Unique Calls')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Plot 6: Scatter plot showing relationship between max_depth and unique_calls
    difficulties = df['difficulty'].unique()
    colors = sns.color_palette("husl", len(difficulties))
    
    for i, difficulty in enumerate(difficulties):
        subset = df[df['difficulty'] == difficulty]
        axes[1, 2].scatter(subset['max_depth'], subset['unique_calls'], 
                          label=difficulty, alpha=0.6, color=colors[i])
    
    axes[1, 2].set_title('Max Length vs Unique Calls by Difficulty')
    axes[1, 2].set_xlabel('Max Length')
    axes[1, 2].set_ylabel('Unique Calls')
    axes[1, 2].legend()
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Create additional correlation plot
    plt.figure(figsize=(10, 6))
    
    # Create separate subplot for each difficulty
    difficulties = sorted(df['difficulty'].unique())
    n_difficulties = len(difficulties)
    
    fig2, axes2 = plt.subplots(1, n_difficulties, figsize=(5*n_difficulties, 5))
    if n_difficulties == 1:
        axes2 = [axes2]
    
    for i, difficulty in enumerate(difficulties):
        subset = df[df['difficulty'] == difficulty]
        axes2[i].scatter(subset['max_depth'], subset['unique_calls'], alpha=0.6)
        axes2[i].set_title(f'Difficulty: {difficulty}')
        axes2[i].set_xlabel('Max Length')
        axes2[i].set_ylabel('Unique Calls')
        
        # Add correlation coefficient
        corr = subset['max_depth'].corr(subset['unique_calls'])
        axes2[i].text(0.05, 0.95, f'Corr: {corr:.3f}', transform=axes2[i].transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Max Length vs Unique Calls Correlation by Difficulty')
    plt.tight_layout()
    plt.show()

def generate_summary_report(df):
    """
    Generate a comprehensive summary report
    """
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 60)
    print(f"Total instances analyzed: {len(df)}")
    print(f"Difficulties present: {sorted(df['difficulty'].unique())}")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Overall statistics
    print("OVERALL STATISTICS:")
    print("-" * 30)
    overall_stats = df[['max_depth', 'unique_calls']].describe()
    print(overall_stats)
    print()
    
    # Statistics by difficulty
    stats, summary_df = analyze_instance_metrics(df)
    
    # Additional insights
    print("ADDITIONAL INSIGHTS:")
    print("-" * 30)
   
    # Find difficulty with highest/lowest averages
    max_depth_highest = summary_df.loc[summary_df['max_depth_mean'].idxmax(), 'difficulty']
    max_depth_lowest = summary_df.loc[summary_df['max_depth_mean'].idxmin(), 'difficulty']
    
    unique_calls_highest = summary_df.loc[summary_df['unique_calls_mean'].idxmax(), 'difficulty']
    unique_calls_lowest = summary_df.loc[summary_df['unique_calls_mean'].idxmin(), 'difficulty']
    
    print(f"Highest average max_depth: {max_depth_highest}")
    print(f"Lowest average max_depth: {max_depth_lowest}")
    print(f"Highest average unique_calls: {unique_calls_highest}")
    print(f"Lowest average unique_calls: {unique_calls_lowest}")
    print()
    
    # Overall correlation
    overall_corr = df['max_depth'].corr(df['unique_calls'])
    print(f"Overall correlation between max_depth and unique_calls: {overall_corr:.3f}")
    print()

#%%


analyze_instance_metrics(
    df,
    name="human_difficulty",
    metrics=[
        'max_depth',
        'unique_calls',
        'snipped_max_depth',
        'snipped_unique_calls',
        'modified_func_size',
    ]
    
)


# generate_summary_report(df)
# create_plots(df)

#%%