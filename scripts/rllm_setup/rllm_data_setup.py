from pathlib import Path

def create_data_registry(data_dir: Path) -> dict[str, dict[str, str]]:
    data_registry_dict = {}
    for dataset_dir in data_dir.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            dataset_registry_dict = {}
            for split_file in dataset_dir.glob("*.parquet"):
                split_name = split_file.stem
                dataset_registry_dict[split_name] = str(split_file.resolve())
            data_registry_dict[dataset_name] = dataset_registry_dict
    return data_registry_dict   


if __name__ == '__main__':
    import sys
    import json

    import rllm

    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <data_dir>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Error: {data_dir} does not exist or is not a directory.")
        sys.exit(1)

    
    data_registry = create_data_registry(data_dir)

    rllm_data_registry_path = Path(rllm.__file__).parent / "registry" / "dataset_registry.json"
    rllm_data_registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rllm_data_registry_path, "w") as f:
        json.dump(data_registry, f, indent=4)

    print(json.dumps(data_registry, indent=4))
