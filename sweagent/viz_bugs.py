import json
from pathlib import Path
import html

def viz_bugs(
   bugs: Path | str,
   output: Path | str,
):
    bugs, output = Path(bugs), Path(output)
    bugs_data = json.load(open(bugs, "r"))

    content = ""
    for d in bugs_data:

        content += f"""<h3>{d['instance_id']}</h3>
    {html.escape(d['problem_statement']).replace('\n', '<br>\n')}
    <hr>
    {html.escape(d['patch']).replace('\n', '<br>\n')}
    <hr>
    """
        

    html_text = f"""<!DOCTYPE html>
    <html>
    <body>

    <h1 style="font-size:60px;">Outputs</h1>

    {content}

    </body>
    </html>"""

    output.write_text(html_text)


if __name__ == '__main__':

    import fire
    fire.Fire(viz_bugs)