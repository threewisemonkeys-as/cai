import logging
from pathlib import Path
import html

from athils import JsonLinesFile

input_file = Path("data/swesmith/db_amenable_wo_fix_o3.jsonl")

data = JsonLinesFile.read_from(input_file)

yeses, nos, nones, errs = [], [], [], []

content = ""

for d in data:
    instance_id = d['instance_id']
    answer = d['selection_answer']

    content += f"""<h3>{d['instance_id']}</h3>
{html.escape(d['selection_response']).replace('\n', '<br>\n')}
"""
    
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

html_text = f"""<!DOCTYPE html>
<html>
<body>

<h1 style="font-size:60px;">Outputs</h1>

{content}

</body>
</html>"""

output_file = input_file.parent / f"{input_file.stem}.html"
output_file.write_text(html_text)
