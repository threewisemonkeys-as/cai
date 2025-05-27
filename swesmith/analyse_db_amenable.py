import logging

from athils import JsonLinesFile

data = JsonLinesFile.read_from("data/swesmith/db_amenable_wo_fix_o3.jsonl")

yeses, nos, nones, errs = [], [], [], []

for d in data:
    instance_id = d['instance_id']
    answer = d['selection_answer']
    
    if answer == True:
        yeses.append(instance_id)
    elif answer == False:
        nos.append(instance_id)
    elif answer == None:
        nones.append(instance_id)
    else:
        logging.error(f"{instance_id} has answer {answer}")
        errs.append(instance_id)

    
print(f"{len(yeses)=}, {len(nos)=}, {len(nones)=}, {len(errs)=}")
