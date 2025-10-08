import json
from pathlib import Path

from datasets import Dataset

def convert(
    output_file: Path | str,
    *input_files: list[Path | str],
):
    output_file = Path(output_file)
    output_data, recorded = [], set()
    for input_file in input_files:
        data = json.load(open(input_file, "r"))
        for instance in data:
            key = (instance['repo'], instance['problem_statement'])
            if key in recorded:
                continue
            output_instance = {**instance}
            output_instance['base_commit'] = 'main'
            output_data.append(output_instance)
            recorded.add(key)
    
    json.dump(output_data, open(output_file, "w"), indent=2)
    print(f"Wrote {len(output_data)} instances to {output_file.resolve()}")


if __name__ == '__main__':
    import fire
    fire.Fire(convert)