import random 
from pathlib import Path

from datasets import load_dataset
from athils import JsonLinesFile

from dotenv import load_dotenv

load_dotenv()  

def filter(
    output_path: str,
    ds_name: str = "SWE-bench/SWE-smith",
    ds_split: str = "train",
    k: int = 200,
    filter_out_empty_ps: bool = True,
):
    ds = load_dataset(ds_name)
    ds = ds[ds_split]

    if filter_out_empty_ps:
        ds = [dd for dd in ds if dd['problem_statement'] != '']
        
    random.seed(42)
    sample = random.sample(ds, k=k)
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    JsonLinesFile.write_to(output_path, sample)

if __name__ == '__main__':
    import fire
    fire.Fire(filter)