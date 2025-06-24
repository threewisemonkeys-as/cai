
from pathlib import Path
import uuid
import json
import logging

STRATEGY_NAME = "sweagent_buggen"

def collect(
    input_dir: Path | str,
    output_file: Path | str,
):
    
    output = []
    for patch_file in Path(input_dir).rglob("*.patch"):

        traj_path = patch_file.parent / f"{patch_file.stem}.traj"
        if not traj_path.exists():
            logging.warning(f"Could not find trajectory at: {traj_path}")
            continue

        if traj_path.read_text().strip() == "":
            logging.warning(f"Empty trajectory file at: {traj_path}")
        
        traj = json.load(open(traj_path, "r"))

        if "trajectory" not in traj:
            logging.warning(f"Could not find trajectory inside: {traj_path}")
        
        traj_data = traj["trajectory"]
        last_action = traj_data[-1]
        if last_action["response"].startswith("Exit due to"):
            logging.info(f"Skipping {traj_path}")
            continue

        patch = patch_file.read_text()
        image_name = patch_file.parent.parent.parent.name.split(':')[0]
        _, _, repo_name, base_commit = image_name.split('.')
        instance_id = f"{repo_name}.{base_commit}.{STRATEGY_NAME}__{str(uuid.uuid4())[:8]}"

        output.append({
            "strategy": STRATEGY_NAME,
            "image_name": image_name,
            "patch": patch,
            "instance_id": instance_id,
        })

    Path(output_file).write_text(json.dumps(output, indent=4))
    print(f"Wrote {len(output)} instances to {output_file}")


if __name__ == '__main__':

    import fire
    fire.Fire(collect)



