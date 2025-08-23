#!/usr/bin/env python3
"""
Script to add 'ds_name' field to JSON datasets based on filename patterns.
Saves the modified data to the same path but with 'ds_name' added to the filename.

Usage:
    python add_ds_name.py <input_json_file>

Mapping rules:
- swesmith* files -> "ds_name": "swe-smith"
- r2egym* files -> "ds_name": "r2egym"
- buggen* files -> "ds_name": "d3"
"""

import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Union


def determine_ds_name(filename: str) -> str:
    """
    Determine the ds_name value based on the filename pattern.
    
    Args:
        filename: The name of the input file
        
    Returns:
        The appropriate ds_name value
        
    Raises:
        ValueError: If filename doesn't match any known pattern
    """
    filename_lower = filename.lower()
    
    if filename_lower.startswith('swesmith'):
        return "swe-smith"
    elif filename_lower.startswith('r2egym'):
        return "r2egym"
    elif filename_lower.startswith('buggen'):
        return "d3"
    else:
        raise ValueError(f"Unknown filename pattern: {filename}. Expected swesmith*, r2egym*, or buggen*")


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


def add_ds_name_to_json_file(input_path: str) -> str:
    """
    Add ds_name field to all data points in a JSON file.
    
    Args:
        input_path: Path to the input JSON file
        
    Returns:
        Path to the output file
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If input file is not valid JSON
        ValueError: If filename doesn't match known patterns
    """
    input_path = Path(input_path)
    
    # Check if input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine ds_name from filename
    ds_name = determine_ds_name(input_path.name)
    output_path = input_path
    
    print(f"Reading data from: {input_path}")
    print(f"Determined ds_name: {ds_name}")
    
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
                transformed_point = add_ds_name_to_data_point(data_point, ds_name)
                transformed_data.append(transformed_point)
                count_transformed += 1
            else:
                # If it's not a dict, keep as is
                transformed_data.append(data_point)
                
        print(f"Added ds_name to {count_transformed} data points out of {len(data)} total")
        
    elif isinstance(data, dict):
        # Handle single data point
        transformed_data = add_ds_name_to_data_point(data, ds_name)
        print(f"Added ds_name to 1 data point")
        
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
        print("Usage: python add_ds_name.py <input_json_file>")
        print("Example: python add_ds_name.py buggen_tasks/splits/swesmith_valid.json")
        print()
        print("Supported filename patterns:")
        print("  swesmith* -> ds_name: 'swe-smith'")
        print("  r2egym*   -> ds_name: 'r2egym'")
        print("  buggen*   -> ds_name: 'd3'")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        output_file = add_ds_name_to_json_file(input_file)
        print(f"Successfully processed file: {input_file}")
        print(f"Output saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
