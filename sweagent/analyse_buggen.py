import json
from pathlib import Path
import logging
from collections import defaultdict

def analyse(
    traj_dir: Path | str,
    agent_dir: Path | str,
):

    agent_results = {}

    for rdir in Path(agent_dir).iterdir():
        agent_results[rdir.name.split(".sweagent_buggen__")[0]] = rdir / "debug_gym.jsonl"

    for patch_path in Path(traj_dir).rglob("*.patch"):
        # traj_path = patch_path.parent / f"{patch_path.stem}.traj"
        # if not traj_path.exists():
        #     logging.warning(f"Could not find trajectory at: {traj_path}")
        #     continue

        # if traj_path.read_text().strip() == "":
        #     logging.warning(f"Empty trajectory file at: {traj_path}")
        
        # traj = json.load(open(traj_path, "r"))

        # if "trajectory" not in traj:
        #     logging.warning(f"Could not find trajectory inside: {traj_path}")
        
        # traj_data = traj["trajectory"]
        # last_action = traj_data[-1]
        # if last_action["response"].startswith("Exit due to"):
        #     logging.info(f"Skipping {traj_path}")
        #     continue


        instance_id = patch_path.parent.parent.name.split("swesmith.x86_64.")[1]

        if instance_id not in agent_results or not agent_results[instance_id].exists():
            logging.warning(f"Could not find {instance_id} data in {agent_dir}")
            continue
    
        agent_data = json.load(open(agent_results[instance_id], "r"))

        tool_counter = defaultdict(lambda: 0)
        for step in agent_data.get("log", []):
            if step.get("action") is None:
                continue
            tool_counter[step["action"]["name"]] += 1
        
        print(f'\n\n{instance_id}\n{tool_counter}')


        


if __name__ == '__main__':
    import fire
    fire.Fire(analyse)
