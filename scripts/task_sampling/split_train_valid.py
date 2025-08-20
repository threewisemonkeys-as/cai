#!/usr/bin/env python3
"""
Script to sample training and validation data from three datasets.
Samples 1000 training and 25 validation data points from each dataset.
Saves each split and each dataset individually into json files.
"""

import json
import random
import os
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

def load_json_dataset(filepath):
    """Load a JSON dataset from file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, filepath):
    """Save data to JSON format."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def sample_train_valid(data, train_size=1000, valid_size=25):
    """
    Sample training and validation data from the dataset.
    
    Args:
        data: List of data items
        train_size: Number of training samples
        valid_size: Number of validation samples
    
    Returns:
        tuple: (train_data, valid_data)
    """
    if len(data) < train_size + valid_size:
        raise ValueError(f"Dataset has only {len(data)} items, but need {train_size + valid_size}")
    
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
        "swesmith": base_dir / "swesmith_2000.json"
    }
    
    print("Processing datasets...")
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        
        # Load the dataset
        data = load_json_dataset(dataset_path)
        print(f"Loaded {len(data)} items from {dataset_name}")
        
        # Sample train and validation data
        try:
            train_data, valid_data = sample_train_valid(data, train_size=1000, valid_size=25)
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
