#!/usr/bin/env python3
"""
Script to sample training and validation data from three datasets.
Samples 1000 training and 25 validation data points from each dataset.
Saves each split and each dataset individually into json files.
Also adds 'ds_name' field to each data point based on the dataset.
Also ensures both 'image_name' and 'docker_image' fields exist in each data point.
"""

import json
import random
import os
from pathlib import Path
from typing import Any, Dict, List

# Set random seed for reproducibility
random.seed(42)

def determine_ds_name(dataset_name: str) -> str:
    """
    Determine the ds_name value based on the dataset name.
    
    Args:
        dataset_name: The name of the dataset
        
    Returns:
        The appropriate ds_name value
        
    Raises:
        ValueError: If dataset name doesn't match any known pattern
    """
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower == 'swesmith':
        return "swe-smith"
    elif dataset_name_lower == 'r2egym':
        return "r2egym"
    elif dataset_name_lower == 'buggen':
        return "d3"
    elif dataset_name_lower == 'featadd':
        return "d4"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Expected swesmith, r2egym, buggen, or featadd")


def add_ds_name_to_data_point(data_point: Dict[str, Any], ds_name: str) -> Dict[str, Any]:
    """
    Add ds_name field to a single data point.
    
    Args:
        data_point: Dictionary representing a single data point
        ds_name: The ds_name value to add
        
    Returns:
        Data point with ds_name field added
    """
    if isinstance(data_point, dict):
        # Create a copy to avoid modifying the original
        transformed = data_point.copy()
        # Add the ds_name field
        transformed['ds_name'] = ds_name
        return transformed
    return data_point


def add_ds_name_to_dataset(data: List[Dict[str, Any]], ds_name: str) -> List[Dict[str, Any]]:
    """
    Add ds_name field to all data points in a dataset.
    
    Args:
        data: List of data points
        ds_name: The ds_name value to add
        
    Returns:
        List of data points with ds_name field added
    """
    transformed_data = []
    for data_point in data:
        if isinstance(data_point, dict):
            transformed_point = add_ds_name_to_data_point(data_point, ds_name)
            transformed_data.append(transformed_point)
        else:
            # If it's not a dict, keep as is
            transformed_data.append(data_point)
    return transformed_data


def ensure_image_fields_in_data_point(data_point: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure both 'image_name' and 'docker_image' fields exist in a data point.
    If one is missing, adds the other with the same value as the existing field.
    
    Args:
        data_point: Dictionary representing a single data point
        
    Returns:
        Data point with both image_name and docker_image fields (if either existed)
    """
    # Create a copy to avoid modifying the original
    transformed = data_point.copy()
    
    has_image_name = 'image_name' in transformed
    has_docker_image = 'docker_image' in transformed
    
    if has_image_name and not has_docker_image:
        # Add docker_image with same value as image_name
        transformed['docker_image'] = transformed['image_name']
    elif has_docker_image and not has_image_name:
        # Add image_name with same value as docker_image
        transformed['image_name'] = transformed['docker_image']
    # If both exist or neither exists, leave unchanged
    
    return transformed


def ensure_image_fields_in_dataset(data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Ensure both 'image_name' and 'docker_image' fields exist in all data points of a dataset.
    
    Args:
        data: List of data points
        
    Returns:
        Tuple of (transformed_data, stats_dict)
    """
    transformed_data = []
    stats = {
        'added_docker_image': 0,
        'added_image_name': 0,
        'both_existed': 0,
        'neither_existed': 0
    }
    
    for data_point in data:
        if isinstance(data_point, dict):
            has_image_name_before = 'image_name' in data_point
            has_docker_image_before = 'docker_image' in data_point
            
            transformed_point = ensure_image_fields_in_data_point(data_point)
            transformed_data.append(transformed_point)
            
            # Count transformations
            if has_image_name_before and not has_docker_image_before:
                stats['added_docker_image'] += 1
            elif has_docker_image_before and not has_image_name_before:
                stats['added_image_name'] += 1
            elif has_image_name_before and has_docker_image_before:
                stats['both_existed'] += 1
            else:
                stats['neither_existed'] += 1
        else:
            # If it's not a dict, keep as is
            transformed_data.append(data_point)
    
    return transformed_data, stats


def load_json_dataset(filepath):
    """Load a JSON dataset from file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, filepath):
    """Save data to JSON format."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def sample_train_valid(data, train_size=1000, valid_size=25, dataset_name=None):
    """
    Sample training and validation data from the dataset.
    
    Args:
        data: List of data items
        train_size: Number of training samples
        valid_size: Number of validation samples
        dataset_name: Name of the dataset (for adding ds_name field)
    
    Returns:
        tuple: (train_data, valid_data)
    """
    if len(data) < train_size + valid_size:
        raise ValueError(f"Dataset has only {len(data)} items, but need {train_size + valid_size}")
    
    # Ensure image fields exist in all data points
    data, image_stats = ensure_image_fields_in_dataset(data)
    print(f"Image field transformations:")
    print(f"  Added 'docker_image' field to {image_stats['added_docker_image']} data points")
    print(f"  Added 'image_name' field to {image_stats['added_image_name']} data points") 
    print(f"  Both fields already existed in {image_stats['both_existed']} data points")
    print(f"  Neither field existed in {image_stats['neither_existed']} data points")
    
    # Add ds_name field to all data points if dataset_name is provided
    if dataset_name:
        ds_name = determine_ds_name(dataset_name)
        data = add_ds_name_to_dataset(data, ds_name)
        print(f"Added ds_name '{ds_name}' to all data points")
    
    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Split into train and validation
    train_data = shuffled_data[:train_size]
    valid_data = shuffled_data[train_size:train_size + valid_size]
    
    return train_data, valid_data

def main():
    # Define paths
    base_dir = Path("/Users/ericyuan/GitHub/cai/scripts/task_sampling/buggen_tasks")
    output_dir = base_dir.parent / "splits"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Dataset files
    datasets = {
        "buggen": base_dir / "buggen_2000.json",
        "r2egym": base_dir / "r2egym_2000.json", 
        "swesmith": base_dir / "swesmith_2000.json",
        "featadd": base_dir / "featadd_1065.json"
    }
    
    print("Processing datasets...")
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        
        # Load the dataset
        data = load_json_dataset(dataset_path)
        print(f"Loaded {len(data)} items from {dataset_name}")
        
        # Sample train and validation data
        try:
            train_data, valid_data = sample_train_valid(data, train_size=1000, valid_size=25, dataset_name=dataset_name)
            print(f"Sampled {len(train_data)} training and {len(valid_data)} validation items")
            
            # Save training data
            train_output_path = output_dir / f"{dataset_name}_train.json"
            save_json(train_data, train_output_path)
            print(f"Saved training data to: {train_output_path}")
            
            # Save validation data
            valid_output_path = output_dir / f"{dataset_name}_valid.json"
            save_json(valid_data, valid_output_path)
            print(f"Saved validation data to: {valid_output_path}")
            
        except ValueError as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    print(f"\nAll datasets processed! Output saved to: {output_dir}")
    
    # Print summary
    print("\nSummary of generated files:")
    for dataset_name in datasets.keys():
        train_file = output_dir / f"{dataset_name}_train.json"
        valid_file = output_dir / f"{dataset_name}_valid.json"
        if train_file.exists() and valid_file.exists():
            print(f"  {dataset_name}: {train_file.name} ({1000} items), {valid_file.name} ({25} items)")

if __name__ == "__main__":
    main()
