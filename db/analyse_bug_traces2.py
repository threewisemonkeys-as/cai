from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from athils import JsonLinesFile
from utils import snip_trace
from trace_viz import render_trace


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


@dataclass
class ComplexityMetrics:
    """Container for trace complexity analysis with statistical summaries"""
    max_depth_stats: Stats = None
    cross_module_stats: Stats = None
    top_module_interactions: List[Tuple[str, int]] = None
    
    def __post_init__(self):
        if self.max_depth_stats is None:
            self.max_depth_stats = Stats()
        if self.cross_module_stats is None:
            self.cross_module_stats = Stats()
        if self.top_module_interactions is None:
            self.top_module_interactions = []


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


def extract_module_name(location: str) -> str:
    """
    Extract module name from a location string.
    
    Examples:
    'sqlparse/sql.py:151' -> 'sqlparse/sql.py'
    'tests/test_cli.py:14' -> 'tests/test_cli.py' 
    '<frozen importlib._bootstrap>:241' -> '<frozen importlib._bootstrap>'
    """
    if ':' in location:
        return location.split(':')[0]
    return location


def calculate_max_call_depth(trace_data: List[Dict]) -> int:
    """
    Calculate the maximum call depth in a trace.
    
    Think of this like measuring how deep a rabbit hole goes - 
    some function calls lead to deeper and deeper nested calls.
    """
    if not trace_data:
        return 0
    
    # The trace data already contains depth information
    depths = [entry.get('depth', 0) for entry in trace_data]
    return max(depths) if depths else 0


def calculate_cross_module_calls(trace_data: List[Dict]) -> Tuple[int, Dict[str, int]]:
    """
    Calculate cross-module call frequency.
    
    Think of this like measuring how often your code jumps between different
    packages/modules - high cross-module activity suggests either good modular
    design or potential coupling issues.
    
    Returns:
        Tuple of (total_cross_module_calls, module_interaction_counts)
    """
    cross_module_count = 0
    module_interactions = defaultdict(int)
    
    for entry in trace_data:
        current_location = entry.get('location', '')
        parent_location = entry.get('parent_location', '')
        
        # Skip if we don't have both locations
        if not current_location or not parent_location:
            continue
            
        current_module = extract_module_name(current_location)
        parent_module = extract_module_name(parent_location)
        
        # Count as cross-module if modules are different
        if current_module != parent_module and current_module and parent_module:
            cross_module_count += 1
            # Track specific module interactions (from -> to)
            interaction = f"{parent_module} -> {current_module}"
            module_interactions[interaction] += 1
    
    return cross_module_count, dict(module_interactions)


def format_instance_results(instance: dict, test_traces: list, repo_stats: Dict[str, Any], threshold: Optional[int] = None, result: dict | None = None) -> str:
    """Format results for a single repository"""
    output = [f"\n{'='*60}"]
    output.append(f"Problem: {instance['instance_id']}")
    output.append('='*60)

    output.append(instance['problem_statement'])
    output.append('-'*40)
    output.append(instance['patch'])
    output.append('-'*40)


    if result is not None:
        output.append(f"Solved by agent: {result['success']}")
        output.append(result['patch'])

    output.append(f"Total test points analyzed: {repo_stats['num_points']}")
    output.append(f"Entrypoint found rate: {repo_stats['ep_found_rate']:.2%}")
    
    if threshold is not None:
        above_threshold = repo_stats['above_threshold_count']
        total_with_ep = repo_stats['total_with_entrypoint']
        percentage = (above_threshold / total_with_ep * 100) if total_with_ep > 0 else 0
        output.append(f"Traces with >{threshold} unique calls: {above_threshold}/{total_with_ep} ({percentage:.1f}%)")
    
    # Add complexity metrics
    complexity = repo_stats.get('complexity_metrics')
    if complexity:
        output.append("")
        output.append("COMPLEXITY ANALYSIS:")
        output.append("Call Depth Distribution:")
        output.append(f"  {complexity.max_depth_stats}")
        output.append("Cross-Module Call Distribution:")
        output.append(f"  {complexity.cross_module_stats}")
        
        if complexity.top_module_interactions:
            output.append("Top module interactions:")
            for interaction, count in complexity.top_module_interactions[:5]:  # Show top 5
                output.append(f"  {interaction}: {count} calls")
    
    output.append("")
    output.append("STATISTICAL BREAKDOWN:")
    output.append("Trace Lengths:")
    output.append(f"  {repo_stats['lengths']}")
    output.append("")
    output.append("Post-Entrypoint Trace Lengths:")
    output.append(f"  {repo_stats['post_ep_lens']}")
    output.append("")
    output.append("Post-Entrypoint Unique Calls:")
    output.append(f"  {repo_stats['post_ep_unique']}")
    output.append("")
    output.append("Post-Entrypoint Max Depth:")
    output.append(f"  {repo_stats['post_ep_max_depth']}")


    output.append('-'*60)
    for test, trace in test_traces:
        output.append(f"TRACE FOR TEST: {test}\n{'-'*10}")
        try:
            snipped = snip_trace(trace['trace_data'], test)
            output.append(render_trace(snipped))
        except:
            pass
        output.append('-'*30)
    
    return "\n".join(output)


def format_summary_results(all_repo_stats: Dict[str, Dict[str, Any]], threshold: Optional[int] = None) -> str:
    """Format summary statistics across all repositories"""
    # Aggregate data across all repos
    all_lengths = []
    all_post_ep_lens = []
    all_post_ep_unique = []
    all_post_ep_max_depth = []
    all_ep_found_rates = []
    total_points = 0
    total_above_threshold = 0
    total_with_entrypoint = 0
    
    # Complexity aggregation
    all_max_depths = []
    all_cross_module_counts = []
    all_module_interactions = defaultdict(int)
    
    for repo_stats, _, _ in all_repo_stats.values():
        all_lengths.extend(repo_stats['raw_lengths'])
        all_post_ep_lens.extend(repo_stats['raw_post_ep_lens'])
        all_post_ep_unique.extend(repo_stats['raw_post_ep_unique'])
        all_post_ep_max_depth.extend(repo_stats['raw_post_ep_max_depth'])
        all_ep_found_rates.append(repo_stats['ep_found_rate'])
        total_points += repo_stats['num_points']
        
        # Complexity metrics - collect individual values from all traces
        all_max_depths.extend(repo_stats['raw_max_depths'])
        all_cross_module_counts.extend(repo_stats['raw_cross_module_counts'])
        
        # Aggregate module interactions
        complexity = repo_stats.get('complexity_metrics')
        if complexity:
            for interaction, count in complexity.top_module_interactions:
                all_module_interactions[interaction] += count
        
        if threshold is not None:
            total_above_threshold += repo_stats['above_threshold_count']
            total_with_entrypoint += repo_stats['total_with_entrypoint']
    
    # Calculate summary stats
    summary_lengths = calculate_stats(all_lengths)
    summary_post_ep_lens = calculate_stats(all_post_ep_lens)
    summary_post_ep_unique = calculate_stats(all_post_ep_unique)
    summary_post_ep_max_depth = calculate_stats(all_post_ep_max_depth)
    avg_ep_found_rate = sum(all_ep_found_rates) / len(all_ep_found_rates) if all_ep_found_rates else 0
    
    # Overall complexity summary
    overall_max_depth_stats = calculate_stats(all_max_depths)
    overall_cross_module_stats = calculate_stats(all_cross_module_counts)
    top_interactions = sorted(all_module_interactions.items(), key=lambda x: x[1], reverse=True)[:10]
    
    output = [f"\n{'='*60}"]
    output.append("SUMMARY ACROSS ALL REPOSITORIES")
    output.append('='*60)
    output.append(f"Total repositories analyzed: {len(all_repo_stats)}")
    output.append(f"Total test points across all repos: {total_points}")
    output.append(f"Average entrypoint found rate: {avg_ep_found_rate:.2%}")
    
    if threshold is not None:
        overall_percentage = (total_above_threshold / total_with_entrypoint * 100) if total_with_entrypoint > 0 else 0
        output.append(f"Overall traces with >{threshold} unique calls: {total_above_threshold}/{total_with_entrypoint} ({overall_percentage:.1f}%)")
    
    # Overall complexity summary
    output.append("")
    output.append("OVERALL COMPLEXITY ANALYSIS:")
    output.append("Call Depth Distribution Across All Traces:")
    output.append(f"  {overall_max_depth_stats}")
    output.append("Cross-Module Call Distribution Across All Traces:")
    output.append(f"  {overall_cross_module_stats}")
    
    if top_interactions:
        output.append("Most frequent module interactions across all repos:")
        for interaction, count in top_interactions[:5]:
            output.append(f"  {interaction}: {count} calls")
    
    output.append("")
    output.append("STATISTICAL BREAKDOWN:")
    output.append("Overall Trace Lengths:")
    output.append(f"  {summary_lengths}")
    output.append("")
    output.append("Overall Post-Entrypoint Trace Lengths:")
    output.append(f"  {summary_post_ep_lens}")
    output.append("")
    output.append("Overall Post-Entrypoint Unique Calls:")
    output.append(f"  {summary_post_ep_unique}")
    output.append("")
    output.append("Overall Post-Entrypoint Max depths:")
    output.append(f"  {summary_post_ep_max_depth}")
    
    return "\n".join(output)


def analyse(data_path: Path | str, output_file: Optional[str] = None, threshold: Optional[int] = None, result: str | Path | None = None):
    """
    Analyze trace data with enhanced statistics including complexity metrics.
    
    Args:
        data_path: Path to the input data file
        output_file: Optional path to save results
        threshold: Optional threshold N to count traces with >N unique calls
    
    Think of this like analyzing test coverage across different codebases:
    - Each repo is like a different project
    - Each test is like a different test case
    - We're measuring how deep our traces go and how successful our analysis is
    - The threshold is like setting a "complexity bar" to identify high-complexity traces
    - Call depth is like measuring the "height" of your execution stack
    - Cross-module calls show how "connected" your codebase components are
    """
    # Enable debug logging
    logging.basicConfig(level=logging.INFO)
    
    data = JsonLinesFile.read_from(data_path)
    logging.info(f"Loaded data with {len(data)} entries")
    
    all_repo_stats = {}

    if result is not None:
        agent_results = {}
        for log_file in Path(result).rglob("*debug_gym.jsonl"):
            agent_patch_file = log_file.parent / "debug_gym.patch"
            if agent_patch_file.exists():
                agent_patch = agent_patch_file.read_text()
            else:
                agent_patch = None
            agent_data = json.load(open(log_file, "r"))
            agent_results[agent_data['problem']] = {
                'success': agent_data['success'],
                'traj_len': len(agent_data['log']),
                'patch': agent_patch,
            }
    else:
        agent_results = None
                
    
    for instance, test_traces in data:
    
        logging.info(f"Processing instance: {instance['instance_id']} with {len(test_traces)} traces")
        
        lengths, ep_founds, post_ep_lens, post_ep_unique, post_ep_max_depth = [], [], [], [], []
        
        # Complexity tracking - collect individual values for statistics
        max_depths = []
        cross_module_counts = []
        all_module_interactions = defaultdict(int)
        for test, test_data in test_traces:
            test_trace = test_data['trace_data']
            lengths.append(len(test_trace))
            
            # Debug: Print trace info
            logging.info(f"Processing test {test} with {len(test_trace)} trace entries")
            
            # Calculate complexity metrics for this trace
            try:
                max_depth = calculate_max_call_depth(test_trace)
                cross_module_calls, module_interactions = calculate_cross_module_calls(test_trace)
                
                logging.info(f"Test {test}: max_depth={max_depth}, cross_module_calls={cross_module_calls}")
                
                # Collect individual values for statistical analysis
                max_depths.append(max_depth)
                cross_module_counts.append(cross_module_calls)
                
                # Aggregate module interactions
                for interaction, count in module_interactions.items():
                    all_module_interactions[interaction] += count
                    
            except Exception as e:
                logging.error(f"Error calculating complexity metrics for {test}: {e}")
                max_depths.append(0)
                cross_module_counts.append(0)
            
            ep_found = False
            try:
                snipped_trace = snip_trace(test_trace, test)
                ep_found = True
                post_ep_len = len(snipped_trace)
                post_ep_unique_count = len(set([(e['location'], e['name']) for e in snipped_trace]))
                
                post_ep_lens.append(post_ep_len)
                post_ep_unique.append(post_ep_unique_count)
                post_ep_max_depth.append(calculate_max_call_depth(snipped_trace))
            except Exception as e:
                logging.warning(f"Could not find entrypoint {test} in trace (instance: {instance['instance_id']})")
            
            ep_founds.append(ep_found)
        
        # Calculate threshold-based metrics if threshold is provided
        above_threshold_count = 0
        total_with_entrypoint = len(post_ep_unique)  # Only traces where entrypoint was found
        
        if threshold is not None:
            above_threshold_count = sum(1 for count in post_ep_unique if count > threshold)
        
        # Prepare complexity metrics with statistical summaries
        top_interactions = sorted(all_module_interactions.items(), key=lambda x: x[1], reverse=True)
        max_depth_stats = calculate_stats(max_depths)
        cross_module_stats = calculate_stats(cross_module_counts)
        
        logging.info(f"Instance {instance['instance_id']} complexity summary: max_depth_stats={max_depth_stats}, cross_module_stats={cross_module_stats}, interactions={len(top_interactions)}")
        
        complexity_metrics = ComplexityMetrics(
            max_depth_stats=max_depth_stats,
            cross_module_stats=cross_module_stats,
            top_module_interactions=top_interactions
        )
        
        # Calculate statistics for this repo
        repo_stats = {
            'num_points': len(lengths),
            'ep_found_rate': calculate_rate(ep_founds),
            'lengths': calculate_stats(lengths),
            'post_ep_lens': calculate_stats(post_ep_lens),
            'post_ep_unique': calculate_stats(post_ep_unique),
            'post_ep_max_depth': calculate_stats(post_ep_max_depth),
            'above_threshold_count': above_threshold_count,
            'total_with_entrypoint': total_with_entrypoint,
            'complexity_metrics': complexity_metrics,
            # Keep raw data for summary calculations
            'raw_lengths': lengths,
            'raw_post_ep_lens': post_ep_lens,
            'raw_post_ep_unique': post_ep_unique,
            'raw_post_ep_max_depth': post_ep_max_depth,
            'raw_max_depths': max_depths,
            'raw_cross_module_counts': cross_module_counts,
        }
        
        all_repo_stats[instance['instance_id']] = repo_stats, instance, test_traces

    # Generate output
    output_lines = []
    
    # Individual repo results
    for repo_name, (repo_stats, instance, test_traces) in all_repo_stats.items():
        if agent_results is not None and repo_name in agent_results:
            _result = agent_results[repo_name]
        else:
            _result = None
        output_lines.append(format_instance_results(instance, test_traces, repo_stats, threshold, _result))
    
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


if __name__ == '__main__':
    import fire
    
    def test_complexity_functions():
        """Test the complexity functions with sample data"""
        sample_trace = [
            {"location": "tests/test_cli.py:14", "parent_location": "python.py:159", "depth": 0, "call_type": "function_call"},
            {"location": "sqlparse/cli.py:32", "parent_location": "tests/test_cli.py:16", "depth": 1, "call_type": "function_call"},
            {"location": "sqlparse/sql.py:151", "parent_location": "sqlparse/cli.py:45", "depth": 2, "call_type": "function_call"},
            {"location": "sqlparse/tokens.py:15", "parent_location": "sqlparse/sql.py:160", "depth": 3, "call_type": "function_call"},
            {"location": "tests/conftest.py:20", "parent_location": "sqlparse/tokens.py:20", "depth": 2, "call_type": "function_call"},
            {"location": "sqlparse/utils.py:45", "parent_location": "tests/conftest.py:25", "depth": 3, "call_type": "function_call"},
        ]
        
        print("Testing complexity functions...")
        
        max_depth = calculate_max_call_depth(sample_trace)
        print(f"Calculated max depth: {max_depth}")
        
        cross_module_calls, interactions = calculate_cross_module_calls(sample_trace)
        print(f"Cross module calls: {cross_module_calls}")
        print(f"Module interactions: {interactions}")
        
        # Test statistics
        sample_depths = [2, 4, 1, 3, 2, 5, 1]
        sample_cross_modules = [3, 7, 2, 4, 3, 8, 1]
        
        depth_stats = calculate_stats(sample_depths)
        cross_module_stats = calculate_stats(sample_cross_modules)
        
        print(f"\nSample depth statistics: {depth_stats}")
        print(f"Sample cross-module statistics: {cross_module_stats}")
        
        return max_depth, cross_module_calls, interactions
    
    fire.Fire({
        "analyse": analyse,
        "test": test_complexity_functions
    })