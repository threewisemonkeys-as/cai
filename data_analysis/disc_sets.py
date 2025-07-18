
#%%

from datasets import load_dataset

swebv_disc = load_dataset("jatinganhotra/SWE-bench_Verified-discriminative")

data = [
    i for items in swebv_disc.values() for i in items
]

#%%

difficulties = set(i['discriminative_difficulty'] for i in data)
subsets = set(i['discriminative_subset'] for i in data)
sets = {}
for subset in subsets:
    for difficulty in difficulties:
        sets[(subset, difficulty)] = [
            i for i in data if i['discriminative_subset'] == subset and i['discriminative_difficulty'] == difficulty
        ]


#%%



#%%