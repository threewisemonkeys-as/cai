# R2E-gym dataset url:
# https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Lite

import os
import random
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from datasets import load_dataset


class R2EGymDatasetSampler:
    """
    A class for creating balanced samples from the R2E-Gym-Lite dataset.
    """
    
    def __init__(self, dataset_name: str = "R2E-Gym/R2E-Gym-Lite", split: str = "train", exclude_folder: Optional[str] = None):
        """
        Initialize the R2E-Gym dataset sampler.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load (train, test, validation)
            exclude_folder: Path to folder containing JSON files with data points to exclude
        """
        self.dataset_name = dataset_name
        self.split = split
        self.exclude_folder = exclude_folder
        self.dataset = None
        self.excluded_ids = set()
        self._load_dataset()
        if exclude_folder:
            self._load_excluded_ids()
    
    def _load_dataset(self):
        """Load the dataset from HuggingFace."""
        print(f"Loading dataset {self.dataset_name} (split={self.split})...")
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        print(f"Dataset loaded: {len(self.dataset)} tasks available")
        
        # Print available fields to help with debugging
        if len(self.dataset) > 0:
            sample_keys = list(self.dataset[0].keys())
            print(f"Available fields: {sample_keys}")
    
    def _load_excluded_ids(self):
        """Load data point IDs from JSON files in the exclude folder."""
        if not self.exclude_folder or not os.path.exists(self.exclude_folder):
            print(f"Warning: Exclude folder '{self.exclude_folder}' does not exist")
            return
        
        exclude_path = Path(self.exclude_folder)
        json_files = list(exclude_path.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in exclude folder: {self.exclude_folder}")
            return
        
        print(f"Loading excluded data points from {len(json_files)} JSON files...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Handle both single object and list of objects
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]
                
                # Extract unique IDs from each item
                for item in items:
                    if isinstance(item, dict):
                        unique_id = self._generate_unique_id(item)
                        if unique_id:
                            self.excluded_ids.add(unique_id)
                            
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(self.excluded_ids)} data points to exclude")
    
    def _generate_unique_id(self, task: Dict) -> str:
        """Generate a unique identifier for a task using docker_image and problem_statement tokens."""
        docker_image = task.get('docker_image', '')
        problem_statement = task.get('problem_statement', '')
        
        if not docker_image or not problem_statement:
            # Fallback to traditional ID fields if docker_image or problem_statement is missing
            for id_field in ['instance_id', 'id', 'task_id', 'problem_id']:
                if id_field in task and task[id_field] is not None:
                    return str(task[id_field])
            
            # Last resort: use entire item as string
            return json.dumps(task, sort_keys=True)
        
        # Tokenize problem_statement and take first 20 tokens
        tokens = problem_statement.split()[:20]
        problem_prefix = ' '.join(tokens)
        
        # Combine docker_image and problem statement prefix
        unique_id = f"{docker_image}||{problem_prefix}"
        return unique_id
    
    def _should_exclude_task(self, task: Dict) -> bool:
        """Check if a task should be excluded based on loaded exclusions."""
        if not self.excluded_ids:
            return False
        
        unique_id = self._generate_unique_id(task)
        return unique_id in self.excluded_ids
    
    def _apply_filters(self, tasks: List[Dict], filters: Dict) -> List[Dict]:
        """Apply filters to the task list and exclude tasks from excluded_ids."""
        filtered_tasks = []
        excluded_count = 0
        
        for task in tasks:
            # Check exclusions first
            if self._should_exclude_task(task):
                excluded_count += 1
                continue
            
            # Then check other filters
            match = True
            for field, value in filters.items():
                if field not in task:
                    match = False
                    break
                
                task_value = task[field]
                if task_value != value:
                    match = False
                    break
            
            if match:
                filtered_tasks.append(task)
        
        if excluded_count > 0:
            print(f"Excluded {excluded_count} tasks based on exclusion list")
        
        return filtered_tasks
    
    def sample_random_tasks(self, 
                           n_samples: int = 10, 
                           seed: Optional[int] = None,
                           filters: Optional[Dict] = None) -> List[Dict]:
        """
        Sample random tasks from the dataset.
        
        Args:
            n_samples: Number of tasks to sample
            seed: Random seed for reproducibility
            filters: Dictionary of field:value pairs to filter tasks
            
        Returns:
            List of sampled task dictionaries
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
        
        # Apply filters if provided
        available_tasks = list(self.dataset)
        if filters or self.excluded_ids:
            if filters is None:
                filters = {}
            available_tasks = self._apply_filters(available_tasks, filters)
        
        # Sample random tasks
        if n_samples > len(available_tasks):
            n_samples = len(available_tasks)
        
        sampled_indices = random.sample(range(len(available_tasks)), n_samples)
        sampled_tasks = [available_tasks[i] for i in sampled_indices]
        
        return sampled_tasks
    
    def save_samples(self, samples: List[Dict], output_path: str) -> None:
        """
        Save sampled tasks to JSON file.
        
        Args:
            samples: List of sampled tasks
            output_path: Path to save the samples
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"Samples saved to: {output_path}")
    
    def create_balanced_sample(self, 
                              n_total: int = 100,
                              balance_by: str = "auto",
                              seed: Optional[int] = None) -> List[Dict]:
        """
        Create a balanced sample across different repositories.
        
        Args:
            n_total: Total number of samples desired
            balance_by: Field to balance by (default: 'auto' to auto-detect)
            seed: Random seed for reproducibility
            
        Returns:
            List of balanced sampled tasks
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded")
        
        if seed is not None:
            random.seed(seed)
        
        # Auto-detect balance field if needed
        available_fields = list(self.dataset[0].keys())
        if balance_by == "auto":
            # Try common field names for repositories
            possible_fields = ['repo', 'repository', 'repo_name', 'project', 'base_commit']
            balance_by = None
            for field in possible_fields:
                if field in available_fields:
                    balance_by = field
                    break
            
            if balance_by is None:
                print(f"Available fields: {available_fields}")
                raise ValueError("Could not auto-detect a suitable field to balance by. Please specify one manually.")
        
        # Validate the balance field exists
        if balance_by not in available_fields:
            print(f"Available fields: {available_fields}")
            raise ValueError(f"Field '{balance_by}' not found in dataset")
        
        # Get unique values for the balance field
        unique_values = set(task.get(balance_by) for task in self.dataset)
        unique_values = [v for v in unique_values if v is not None]
        
        samples_per_category = max(1, n_total // len(unique_values))
        remainder = n_total % len(unique_values)
        
        print(f"Balancing by field: '{balance_by}'")
        print(f"Creating balanced sample across {len(unique_values)} categories")
        print(f"Base samples per category: {samples_per_category}")
        if remainder > 0:
            print(f"Extra samples for first {remainder} categories: 1 each")
        
        all_samples = []
        total_requested = 0
        total_actual = 0
        
        for i, category in enumerate(unique_values):
            # Add extra sample to some categories for remainder
            n_samples = samples_per_category + (1 if i < remainder else 0)
            total_requested += n_samples
            
            filters = {balance_by: category}
            try:
                category_samples = self.sample_random_tasks(
                    n_samples=n_samples,
                    filters=filters,
                    seed=seed
                )
                all_samples.extend(category_samples)
                total_actual += len(category_samples)
                
                if len(category_samples) < n_samples:
                    print(f"  {category}: {len(category_samples)} samples (requested {n_samples}, limited by available data)")
                else:
                    print(f"  {category}: {len(category_samples)} samples")
            except Exception as e:
                print(f"  Error sampling from {category}: {e}")
        
        print(f"Total requested: {total_requested}, Total actual: {total_actual}")
        
        # If we didn't get enough samples, fill the remainder with random sampling
        if len(all_samples) < n_total:
            shortage = n_total - len(all_samples)
            print(f"Shortage of {shortage} samples. Filling with additional random samples...")
            
            # Get all sample IDs we already have to avoid duplicates
            existing_ids = set()
            id_field = None
            
            # Try to find an ID field to avoid duplicates
            for field in ['instance_id', 'id', 'task_id', 'problem_id']:
                if field in available_fields:
                    id_field = field
                    existing_ids = {task.get(id_field) for task in all_samples}
                    break
            
            # Sample additional tasks, avoiding duplicates if possible
            remaining_tasks = list(self.dataset)
            if id_field:
                remaining_tasks = [task for task in remaining_tasks 
                                 if task.get(id_field) not in existing_ids]
            
            # Also exclude tasks from the exclusion list
            if self.excluded_ids:
                remaining_tasks = [task for task in remaining_tasks 
                                 if not self._should_exclude_task(task)]
            
            if len(remaining_tasks) >= shortage:
                additional_samples = self.sample_random_tasks(
                    n_samples=shortage,
                    seed=seed + 1000 if seed else None  # Different seed for additional samples
                )
                
                # Filter out any duplicates that might still exist
                if id_field:
                    additional_samples = [task for task in additional_samples 
                                        if task.get(id_field) not in existing_ids]
                
                all_samples.extend(additional_samples[:shortage])
                print(f"Added {len(additional_samples[:shortage])} additional random samples")
            else:
                print(f"Warning: Only {len(remaining_tasks)} unique tasks remaining. Cannot reach {n_total} samples.")
                if remaining_tasks:
                    all_samples.extend(remaining_tasks)
        
        # Shuffle the final list
        random.shuffle(all_samples)
        print(f"Final total samples: {len(all_samples)}")
        
        return all_samples


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="R2E-Gym dataset sampler - Create a balanced sample from the R2E-Gym-Lite dataset"
    )
    
    parser.add_argument(
        "num_tasks", 
        type=int,
        help="Number of tasks to sample"
    )
    
    parser.add_argument(
        "output_path", 
        type=str,
        help="Path to save the output JSON file"
    )
    
    parser.add_argument(
        "--exclude_folder",
        type=str,
        help="Path to folder containing JSON files with R2E-Gym data points to exclude from sampling"
    )
    
    return parser.parse_args()


def main():
    """Main function for the R2E-Gym dataset sampler."""
    args = parse_args()
    
    # Initialize sampler
    sampler = R2EGymDatasetSampler(exclude_folder=args.exclude_folder)
    
    # Create balanced sample
    samples = sampler.create_balanced_sample(
        n_total=args.num_tasks, 
        balance_by="auto", 
        seed=42
    )
    
    # Save samples
    sampler.save_samples(samples, args.output_path)
    
    print(f"\n=== Sampling Complete ===")
    print(f"Total samples: {len(samples)}")
    print(f"Output saved to: {args.output_path}")
    if args.exclude_folder:
        print(f"Excluded data points from folder: {args.exclude_folder}")


if __name__ == "__main__":
    main()