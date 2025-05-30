if __name__ == '__main__':
    from pathlib import Path
    import sys

    from datasets import load_dataset

    output_path = Path(sys.argv[1])

    ds = load_dataset("SWE-bench/SWE-smith")
    image_names = list(sorted(set(ds['train']['image_name'])))
    output_path.write_text('\n'.join(image_names))



