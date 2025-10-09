# Task Sampling Scripts

This directory contains scripts for sampling and processing tasks from various coding benchmark datasets: SWE-smith, R2E-Gym, and other datasets (buggen/featadd).

## Overview

The workflow consists of three main steps:
1. **Sample datasets** - Extract balanced samples from large datasets
2. **Split data** - Create training and validation splits 
3. **Process outputs** - Generate final task files with proper formatting

## Scripts

### 1. `sample_swesmith_dataset.py`

Samples diverse tasks from the SWE-smith dataset with balanced representation across repositories and bug types.

**Usage:**
```bash
python sample_swesmith_dataset.py --num_tasks 2000 --output swesmith_2000.json
```

**Options:**
- `--num_tasks`: Number of tasks to sample (default: 2000)
- `--max_per_combo`: Max samples per repo-bug_type combination (default: 3)
- `--output`: Output file path (default: `swesmith_{num_tasks}.json`)

**Features:**
- Filters out problematic instances and empty problem statements
- Balances across bug types: `pr`, `lm_rewrite`, `combine_file`, `combine_module`, `func_pm_*`
- Ensures diversity across repositories and bug types
- Provides detailed sampling statistics

### 2. `sample_r2egym_dataset.py`

Creates balanced samples from the R2E-Gym-Lite dataset.

**Usage:**
```bash
python sample_r2egym_dataset.py 2000 r2egym_2000.json
```

**Arguments:**
- `num_tasks`: Number of tasks to sample
- `output_path`: Path to save the output JSON file

**Features:**
- Auto-detects repository field for balanced sampling
- Creates balanced distribution across repositories
- Handles cases where categories have limited data
- Supports custom filtering and random seed setting

### 3. `split_train_valid.py`

Processes sampled datasets to create training and validation splits with proper formatting.

**Usage:**
```bash
python split_train_valid.py
```

**Features:**
- Splits each dataset into 1000 training + 25 validation samples
- Adds `ds_name` field to identify source dataset:
  - `swesmith` → `"swe-smith"`
  - `r2egym` → `"r2egym"`  
  - `buggen` → `"d3"`
  - `featadd` → `"d4"`
- Ensures both `image_name` and `docker_image` fields exist
- Saves individual train/valid files for each dataset

## Data Flow

```
Raw Datasets
    ↓
[sample_*_dataset.py] → Balanced samples (e.g., swesmith_2000.json)
    ↓
[split_train_valid.py] → Training/validation splits
    ↓
Final task files in buggen_tasks/tasks/
```

## Output Structure

After running all scripts, you'll have:

```
buggen_tasks/
├── buggen_2000.json          # Raw sampled data
├── featadd_1065.json
├── r2egym_2000.json  
├── swesmith_2000.json
└── tasks/                     # Processed splits
    ├── buggen_train.json      # 1000 training tasks
    ├── buggen_valid.json      # 25 validation tasks
    ├── featadd_train.json
    ├── featadd_valid.json
    ├── r2egym_train.json
    ├── r2egym_valid.json
    ├── swesmith_train.json
    └── swesmith_valid.json
```

## Quick Start

To regenerate all task samples and splits:

```bash
# Sample from each dataset
python sample_swesmith_dataset.py --num_tasks 2000 --output buggen_tasks/swesmith_2000.json
python sample_r2egym_dataset.py 2000 buggen_tasks/r2egym_2000.json

# Split into train/validation sets
python split_train_valid.py
```