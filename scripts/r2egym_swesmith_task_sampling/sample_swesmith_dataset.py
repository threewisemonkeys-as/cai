import datasets
import pandas as pd
import numpy as np
import argparse
import json

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Sample diverse SWE-smith tasks (simple approach)')
    parser.add_argument('--num_tasks', type=int, default=2000, 
                       help='Number of tasks to sample (default: 2000)')
    parser.add_argument('--max_per_combo', type=int, default=3,
                       help='Max samples per repo-bug_type combination (default: 3)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: swesmith_{num_tasks}.json)')
    args = parser.parse_args()
    
    target_samples = args.num_tasks
    max_per_repo_bug_combo = args.max_per_combo
    output_path = args.output or f"swesmith_{target_samples}.json"
    
    print(f"Sampling {target_samples} tasks with max {max_per_repo_bug_combo} per repo-bug_type combo...")
    print("Loading SWE-smith dataset...")
    ds = datasets.load_dataset("SWE-bench/SWE-smith", revision="699b53400d3855206a0fbf3ff4beaf1a52f4f232")["train"]

    # Create a dataframe
    df = pd.DataFrame({
        "index": range(len(ds)),
        "instance_id": ds["instance_id"],
        "repo": ds["repo"],
        "patch": ds["patch"],
        "FAIL_TO_PASS": ds["FAIL_TO_PASS"],
        "PASS_TO_PASS": ds["PASS_TO_PASS"],
        "created_at": ds["created_at"],
        "image_name": ds["image_name"],
        "base_commit": ds["base_commit"],
        "problem_statement": ds["problem_statement"]
    })
    print(f'Original number of instances: {len(df)}')

    # Filter out problematic instances
    problematic_prefixes = [
        "spulec__freezegun.5f171db0",
        "facebookresearch__hydra.0f03eb60", 
        "conan-io__conan.86f29e13",
        "life4__textdistance.c3aca916.combine_file__0sfget5n"
    ]
    
    for prefix in problematic_prefixes:
        df = df[~df["instance_id"].str.startswith(prefix)]
    
    print(f'After removing problematic instances: {len(df)}')

    # Filter out empty problem statements
    df = df[df["problem_statement"].str.strip() != ""]
    print(f'After filtering empty problem statements: {len(df)}')

    # Extract bug type
    df["bug_type"] = df["instance_id"].str.split(".").str[-1]
    
    bug_categories = [
        "pr", "lm_rewrite", "combine_file", "combine_module",
        "func_pm_op", "func_pm_class", "func_pm_ctrl", "func_pm_remove"
    ]
    
    df = df[df["bug_type"].str.startswith(tuple(bug_categories))]
    df["bug_type"] = df["bug_type"].apply(
        lambda x: next((cat for cat in bug_categories if x.startswith(cat)), None)
    )
    
    print(f'After bug type filtering: {len(df)}')
    print(f'Bug types available: {sorted(df["bug_type"].unique())}')
    print(f'Repos available: {df["repo"].nunique()}')

    # Simple diverse sampling approach:
    # 1. Group by repo and bug_type, sample up to N instances per group
    # 2. If we need more samples, do random sampling from remaining
    
    # Use the target_samples variable instead of hardcoded value
    # max_per_repo_bug_combo = 3  # Max samples per repo-bug_type combination
    
    # Sample up to max_per_repo_bug_combo from each repo-bug_type combination
    sampled_groups = []
    for (repo, bug_type), group in df.groupby(['repo', 'bug_type']):
        n_samples = min(len(group), max_per_repo_bug_combo)
        sampled = group.sample(n=n_samples, random_state=42)
        sampled_groups.append(sampled)
    
    sampled_df = pd.concat(sampled_groups, ignore_index=True)
    print(f'After grouped sampling: {len(sampled_df)} samples')
    
    # If we have more than target, randomly sample down
    if len(sampled_df) > target_samples:
        sampled_df = sampled_df.sample(n=target_samples, random_state=42)
        print(f'Randomly sampled down to: {len(sampled_df)} samples')
    
    # If we have fewer than target, add more samples ensuring diversity
    elif len(sampled_df) < target_samples:
        remaining_needed = target_samples - len(sampled_df)
        used_indices = set(sampled_df.index)
        remaining_df = df[~df.index.isin(used_indices)]
        
        if len(remaining_df) > 0:
            # Prioritize underrepresented repo-bug_type combinations
            sampled_counts = sampled_df.groupby(['repo', 'bug_type']).size()
            
            def get_priority(row):
                combo_count = sampled_counts.get((row['repo'], row['bug_type']), 0)
                return 1.0 / (combo_count + 1)  # Higher priority for less represented combos
            
            remaining_df = remaining_df.copy()
            remaining_df['priority'] = remaining_df.apply(get_priority, axis=1)
            
            # Sample based on priority weights
            additional_needed = min(remaining_needed, len(remaining_df))
            weights = remaining_df['priority'] / remaining_df['priority'].sum()
            additional_samples = remaining_df.sample(
                n=additional_needed, 
                weights=weights, 
                random_state=42
            )
            additional_samples = additional_samples.drop('priority', axis=1)
            
            sampled_df = pd.concat([sampled_df, additional_samples], ignore_index=True)
            print(f'Added {additional_needed} more samples. Final size: {len(sampled_df)}')

    # Print final statistics
    print(f'\n=== FINAL SAMPLE STATISTICS ===')
    print(f'Total samples: {len(sampled_df)}')
    
    print(f'\nBug type distribution:')
    bug_dist = sampled_df['bug_type'].value_counts()
    for bug_type, count in bug_dist.items():
        percentage = count / len(sampled_df) * 100
        print(f'  {bug_type}: {count} ({percentage:.1f}%)')
    
    print(f'\nRepo diversity:')
    print(f'  Unique repos: {sampled_df["repo"].nunique()}')
    print(f'  Avg samples per repo: {len(sampled_df) / sampled_df["repo"].nunique():.2f}')
    
    repo_dist = sampled_df['repo'].value_counts()
    print(f'  Top 10 repos by sample count:')
    for repo, count in repo_dist.head(10).items():
        print(f'    {repo}: {count}')
    
    print(f'\nRepo-BugType combinations:')
    combo_dist = sampled_df.groupby(['repo', 'bug_type']).size()
    print(f'  Unique repo-bug_type combinations: {len(combo_dist)}')
    print(f'  Avg samples per combination: {len(sampled_df) / len(combo_dist):.2f}')
    
    # Save results
    # Convert to JSON format matching the reference structure
    json_data = []
    for _, row in sampled_df.iterrows():
        json_data.append({
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "patch": row["patch"],
            "FAIL_TO_PASS": row["FAIL_TO_PASS"],
            "PASS_TO_PASS": row["PASS_TO_PASS"],
            "created_at": row["created_at"],
            "image_name": row["image_name"],
            "problem_statement": row["problem_statement"]
        })
    
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'\nSaved to: {output_path}')
    
    # No longer saving instance IDs separately or as CSV
    return sampled_df

if __name__ == "__main__":
    sampled_df = main()
