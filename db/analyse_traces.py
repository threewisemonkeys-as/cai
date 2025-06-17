from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from athils import JsonLinesFile
from utils import snip_trace


@dataclass
class Stats:
    """Container for min/max/avg statistics"""
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    avg_val: Optional[float] = None
    count: int = 0
    
    def __str__(self) -> str:
        if self.count == 0:
            return "No data"
        return f"Min: {self.min_val:.2f}, Max: {self.max_val:.2f}, Avg: {self.avg_val:.2f}"


def calculate_stats(data: List[float]) -> Stats:
    """Calculate min, max, avg statistics for a list of numbers"""
    if not data:
        return Stats()
    
    return Stats(
        min_val=min(data),
        max_val=max(data), 
        avg_val=sum(data) / len(data),
        count=len(data)
    )


def calculate_rate(boolean_data: List[bool]) -> float:
    """Calculate success rate from boolean data"""
    if not boolean_data:
        return 0.0
    return sum(boolean_data) / len(boolean_data)


def format_repo_results(repo_name: str, repo_stats: Dict[str, Any], threshold: Optional[int] = None) -> str:
    """Format results for a single repository"""
    output = [f"\n{'='*60}"]
    output.append(f"Repository: {repo_name}")
    output.append('='*60)
    output.append(f"Total test points analyzed: {repo_stats['num_points']}")
    output.append(f"Entrypoint found rate: {repo_stats['ep_found_rate']:.2%}")
    
    if threshold is not None:
        above_threshold = repo_stats['above_threshold_count']
        total_with_ep = repo_stats['total_with_entrypoint']
        percentage = (above_threshold / total_with_ep * 100) if total_with_ep > 0 else 0
        output.append(f"Traces with >{threshold} unique calls: {above_threshold}/{total_with_ep} ({percentage:.1f}%)")
    
    output.append("")
    output.append("Trace Lengths:")
    output.append(f"  {repo_stats['lengths']}")
    output.append("")
    output.append("Post-Entrypoint Trace Lengths:")
    output.append(f"  {repo_stats['post_ep_lens']}")
    output.append("")
    output.append("Post-Entrypoint Unique Calls:")
    output.append(f"  {repo_stats['post_ep_unique']}")
    
    return "\n".join(output)


def format_summary_results(all_repo_stats: Dict[str, Dict[str, Any]], threshold: Optional[int] = None) -> str:
    """Format summary statistics across all repositories"""
    # Aggregate data across all repos
    all_lengths = []
    all_post_ep_lens = []
    all_post_ep_unique = []
    all_ep_found_rates = []
    total_points = 0
    total_above_threshold = 0
    total_with_entrypoint = 0
    
    for repo_stats in all_repo_stats.values():
        all_lengths.extend(repo_stats['raw_lengths'])
        all_post_ep_lens.extend(repo_stats['raw_post_ep_lens'])
        all_post_ep_unique.extend(repo_stats['raw_post_ep_unique'])
        all_ep_found_rates.append(repo_stats['ep_found_rate'])
        total_points += repo_stats['num_points']
        
        if threshold is not None:
            total_above_threshold += repo_stats['above_threshold_count']
            total_with_entrypoint += repo_stats['total_with_entrypoint']
    
    # Calculate summary stats
    summary_lengths = calculate_stats(all_lengths)
    summary_post_ep_lens = calculate_stats(all_post_ep_lens)
    summary_post_ep_unique = calculate_stats(all_post_ep_unique)
    avg_ep_found_rate = sum(all_ep_found_rates) / len(all_ep_found_rates) if all_ep_found_rates else 0
    
    output = [f"\n{'='*60}"]
    output.append("SUMMARY ACROSS ALL REPOSITORIES")
    output.append('='*60)
    output.append(f"Total repositories analyzed: {len(all_repo_stats)}")
    output.append(f"Total test points across all repos: {total_points}")
    output.append(f"Average entrypoint found rate: {avg_ep_found_rate:.2%}")
    
    if threshold is not None:
        overall_percentage = (total_above_threshold / total_with_entrypoint * 100) if total_with_entrypoint > 0 else 0
        output.append(f"Overall traces with >{threshold} unique calls: {total_above_threshold}/{total_with_entrypoint} ({overall_percentage:.1f}%)")
    
    output.append("")
    output.append("Overall Trace Lengths:")
    output.append(f"  {summary_lengths}")
    output.append("")
    output.append("Overall Post-Entrypoint Trace Lengths:")
    output.append(f"  {summary_post_ep_lens}")
    output.append("")
    output.append("Overall Post-Entrypoint Unique Calls:")
    output.append(f"  {summary_post_ep_unique}")
    
    return "\n".join(output)


def analyse(data_path: Path | str, output_file: Optional[str] = None, threshold: Optional[int] = None):
    """
    Analyze trace data with enhanced statistics.
    
    Args:
        data_path: Path to the input data file
        output_file: Optional path to save results
        threshold: Optional threshold N to count traces with >N unique calls
    
    Think of this like analyzing test coverage across different codebases:
    - Each repo is like a different project
    - Each test is like a different test case
    - We're measuring how deep our traces go and how successful our analysis is
    - The threshold is like setting a "complexity bar" to identify high-complexity traces
    """
    data = JsonLinesFile.read_from(data_path)
    
    all_repo_stats = {}
    
    for repo, repo_data in data:
        lengths, ep_founds, post_ep_lens, post_ep_unique = [], [], [], []
        
        for test, test_data in repo_data:
            test_trace = test_data['trace_data']
            lengths.append(len(test_trace))
            
            ep_found = False
            try:
                snipped_trace = snip_trace(test_trace, test)
                ep_found = True
                post_ep_len = len(snipped_trace)
                post_ep_unique_count = len(set([(e['location'], e['name']) for e in snipped_trace]))
                
                post_ep_lens.append(post_ep_len)
                post_ep_unique.append(post_ep_unique_count)
            except Exception as e:
                logging.warning(f"Could not find entrypoint {test} in trace (repo: {repo})")
            
            ep_founds.append(ep_found)
        
        # Calculate threshold-based metrics if threshold is provided
        above_threshold_count = 0
        total_with_entrypoint = len(post_ep_unique)  # Only traces where entrypoint was found
        
        if threshold is not None:
            above_threshold_count = sum(1 for count in post_ep_unique if count >= threshold)
        
        # Calculate statistics for this repo
        repo_stats = {
            'num_points': len(lengths),
            'ep_found_rate': calculate_rate(ep_founds),
            'lengths': calculate_stats(lengths),
            'post_ep_lens': calculate_stats(post_ep_lens),
            'post_ep_unique': calculate_stats(post_ep_unique),
            'above_threshold_count': above_threshold_count,
            'total_with_entrypoint': total_with_entrypoint,
            # Keep raw data for summary calculations
            'raw_lengths': lengths,
            'raw_post_ep_lens': post_ep_lens,
            'raw_post_ep_unique': post_ep_unique,
        }
        
        all_repo_stats[repo] = repo_stats

    # Generate output
    output_lines = []
    
    # Individual repo results
    for repo_name, repo_stats in all_repo_stats.items():
        output_lines.append(format_repo_results(repo_name, repo_stats, threshold))
    
    # Summary results
    output_lines.append(format_summary_results(all_repo_stats, threshold))
    
    final_output = "\n".join(output_lines)
    
    # Print to console
    print(final_output)
    
    # Optionally save to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write(final_output)
        print(f"\nResults also saved to: {output_file}")
    
    # Print threshold usage info if provided
    if threshold is not None:
        print(f"\nAnalysis completed with complexity threshold: >{threshold} unique calls")
    else:
        print("\nTip: Use --threshold=N to count traces with >N unique calls")


def choose_above_threshold(data_path: Path | str, output_file: str, threshold: int):
    data = JsonLinesFile.read_from(data_path)
    
    results = []
    
    for repo, repo_data in data:
        repo_results = []

        for test, test_data in repo_data:
            test_trace = test_data['trace_data']
            
            try:
                snipped_trace = snip_trace(test_trace, test)
                post_ep_unique_count = len(set([(e['location'], e['name']) for e in snipped_trace]))
                
                if post_ep_unique_count >= threshold:
                    repo_results.append((test, test_data))

            except Exception as e:
                logging.warning(f"Could not find entrypoint {test} in trace (repo: {repo})")

        results.append((repo, repo_results))
    
    results = [(repo, repo_results) for (repo, repo_results) in results if len(repo_results) > 0]

    JsonLinesFile.write_to(output_file, results)
    print(f"Wrote {sum(len(d) for _, d in results)} total traces to {output_file}")


if __name__ == '__main__':
    import fire
    fire.Fire({
        "analyse": analyse,
        "choose": choose_above_threshold,
    })