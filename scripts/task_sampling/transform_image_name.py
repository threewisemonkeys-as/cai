#!/usr/bin/env python3
"""
Script to transform JSON files by renaming 'image_name' attribute to 'docker_image'.
Saves the modified data to the same path but with 'mapkey' in the filename.

Usage:
    python transform_image_name.py <input_json_file>
"""

import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Union


def transform_data_point(data_point: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single data point by renaming 'image_name' to 'docker_image'.
    
    Args:
        data_point: Dictionary representing a single data point
        
    Returns:
        Transformed data point with renamed attribute
    """
    if 'image_name' in data_point:
        # Create a copy to avoid modifying the original
        transformed = data_point.copy()
        # Rename the attribute
        transformed['docker_image'] = transformed.pop('image_name')
        return transformed
    return data_point


def transform_json_file(input_path: str) -> str:
    """
    Transform a JSON file by renaming 'image_name' attributes to 'docker_image'.
    
    Args:
        input_path: Path to the input JSON file
        
    Returns:
        Path to the output file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If input file is not valid JSON
    """
    input_path = Path(input_path)
    
    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Generate output filename with 'mapkey'
    stem = input_path.stem
    suffix = input_path.suffix
    output_filename = f"{stem}_mapkey{suffix}"
    output_path = input_path.parent / output_filename
    
    print(f"Reading data from: {input_path}")
    
    # Load JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Transform the data
    if isinstance(data, list):
        # Handle list of data points
        transformed_data = []
        count_transformed = 0
        
        for i, data_point in enumerate(data):
            if isinstance(data_point, dict):
                original_keys = set(data_point.keys())
                transformed_point = transform_data_point(data_point)
                transformed_data.append(transformed_point)
                
                # Count transformations
                if 'image_name' in original_keys:
                    count_transformed += 1
            else:
                # If it's not a dict, keep as is
                transformed_data.append(data_point)
                
        print(f"Transformed {count_transformed} data points out of {len(data)} total")
        
    elif isinstance(data, dict):
        # Handle single data point
        original_keys = set(data.keys())
        transformed_data = transform_data_point(data)
        
        count_transformed = 1 if 'image_name' in original_keys else 0
        print(f"Transformed {count_transformed} data point")
        
    else:
        # Handle other data types (shouldn't happen for typical JSON files)
        transformed_data = data
        print("No transformation needed - data is not a dict or list of dicts")
    
    # Save transformed data
    print(f"Saving transformed data to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=2, ensure_ascii=False)
    
    return str(output_path)


def main():
    """Main function to handle command line arguments and execute transformation."""
    if len(sys.argv) != 2:
        print("Usage: python transform_image_name.py <input_json_file>")
        print("Example: python transform_image_name.py data/swesmith_valid.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        output_file = transform_json_file(input_file)
        print(f"Successfully transformed file: {input_file}")
        print(f"Output saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
