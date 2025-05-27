import random 

from datasets import load_dataset
from athils import JsonLinesFile

from dotenv import load_dotenv

load_dotenv()  

ds = load_dataset("SWE-bench/SWE-smith")


non_empty_desc = [dd for dd in ds['train'] if dd['problem_statement'] != '']
random.seed(42)
sample = random.sample(non_empty_desc, k=200)
JsonLinesFile.write_to("data/swesmith/subset.jsonl", sample)
