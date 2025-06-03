from pathlib import Path
import json


def collect(
    input_data: Path | str,
):
    input_data = Path(input_data)
    data = [json.loads(d) for d in input_data.read_text().splitlines()]
    
    perf = sum(d['reward'] for d in data) / len(data)
    print(f"{perf=:.4f}")
    
    

if __name__ == '__main__':
    import fire
    fire.Fire(collect)
