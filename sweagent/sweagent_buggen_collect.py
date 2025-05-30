
from pathlib import Path
import uuid
import json

STRATEGY_NAME = "sweagent_buggen"

def collect(
    input_dir: Path | str,
    output_file: Path | str,
):
    
    output = []
    for patch_file in Path(input_dir).rglob("*.patch"):
        patch = patch_file.read_text()
        image_name = patch_file.parent.parent.name
        _, _, repo_name, base_commit = image_name.split('.')
        instance_id = f"{repo_name}.{base_commit}.{STRATEGY_NAME}__{str(uuid.uuid4())[:8]}"

        output.append({
            "strategy": STRATEGY_NAME,
            "image_name": image_name,
            "patch": patch,
            "instance_id": instance_id,
        })

    Path(output_file).write_text(json.dumps(output, indent=4))


if __name__ == '__main__':

    import fire
    fire.Fire(collect)



