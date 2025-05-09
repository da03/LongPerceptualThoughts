import os
import fire
import json

from tqdm import tqdm
from datetime import datetime
from jinja2.sandbox import SandboxedEnvironment
from pathlib import Path

from stage_1_mcq_gen import generate_mcq_from_captions
from stage_2_simple_cot import generate_simple_cot, collect_simple_cot
from stage_3_expand_cot import generate_extended_cot, collect_extended_cot, create_sft_dpo_dataset


def submit_slurm_jobs():
    
    commands = []
    # prefix = "simple_cot"
    # step = 1000
    # for i in range(0, 69017, step):
    #     commands.append(f"python main.py generate_simple_cot {i} {i + step}")
    
    # step = 1000
    prefix = "expand_cot"
    # for i in range(0, 670072, step):
    #     for cognitive_phrase in ["|Wait,|", "|Let's double-check the details.|", "|Hmm,|", "|Alternatively,|"]:
    #         commands.append(f"""python main.py generate_extended_cot "{cognitive_phrase}" {i} {i + step}""")
    commands = open("commands.txt").read().split("\n")
    
    # avaliable_qos = ["deadline", "deadline", "deadline", "normal", "m", "m", "m2", "m2", "m2", "m3", "m3", "m3", "m3", "m3", "m4", "m4", "m4", "m4", "m4", "m4", "m4", "m4"]
    avaliable_qos = ["m2", "m2", "m2", "m3", "m3", "m3", "m3", "m3", "m4", "m4", "m4", "m4", "m4", "m4", "m4", "m4"]
    for i, command in enumerate(commands):
        qos = avaliable_qos[i % len(avaliable_qos)]
        now = datetime.now()
        work_dir = Path("outputs/slurm_jobs") / f"{now.year}-{now.month}-{now.day}" / f"{now.hour}-{now.minute}-{now.second}-{now.microsecond}"
        output_dir, script_dir, log_dir = create_slrm_job_directory(work_dir)
        job_run_name = f"{prefix}_{i}"
        
        open(script_dir / "run.sh", "w").write(command)
        create_slurm_script(
            job_name=job_run_name,
            log_dir=log_dir,
            script_dir=script_dir, 
            duration=min(partition_to_max_time[qos], 2),
            partition="rtx6000",
            additional_slurm_args = "#SBATCH --account=deadline" if qos == "deadline" else "",
            n_gpus=4,
            qos=qos,
        )
        command = f"sbatch {script_dir / 'launch_job.slrm'}"
        # print(command)
        os.system(command)
        

def update_llamafactory_dataset_info():
    sft_dataset_info = {}
    for p in tqdm(Path("outputs").glob("sft_*.json"), desc="Creating SFT dataset info"):
        
        if "images" in json.load(open(p))[0]:
            
            
            info = {
                "file_name": str(p.absolute()),
                "formatting": "sharegpt",
                "columns": {
                    "messages": "messages", 
                    "images": "images"
                },
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant", 
                    "system_tag": "system"
                }
            }
            if "assistant_prefix" in json.load(open(p))[0]:
                info["columns"]["assistant_prefix"] = "assistant_prefix"
        else:
            info = {
                "file_name": str(p.absolute()),
                "formatting": "sharegpt",
                "columns": {
                    "messages": "messages", 
                },
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant", 
                    "system_tag": "system"
                }
            }
            
        sft_dataset_info[f"long_perceptual_thoughts/{p.stem}"] = info
    json.dump(sft_dataset_info, open("long_perceptual_thoughts_dataset_info.json", "w"), indent=4)
    
    dpo_dataset_info = {}
    for p in tqdm(Path("outputs").glob("dpo_*.json"), desc="Creating DPO dataset info"):
        info = {
            "file_name": str(p.absolute()),
            "formatting": "sharegpt",
            "ranking": True, 
            "columns": {
                "messages": "messages", 
                "images": "images", 
                "chosen": "chosen", 
                "rejected": "rejected"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant", 
                "system_tag": "system"
            }
        }
        dpo_dataset_info[f"long_perceptual_thoughts/{p.stem}"] = info
    json.dump(dpo_dataset_info, open("long_perceptual_thoughts_preference_dataset_info.json", "w"), indent=4)
    
    # Merge it to `data/dataset_info.json`
    orig_dataset_info = json.load(open(os.path.join(os.environ["LLAMAFACTORY_DIR"], "data/dataset_info.json")))
    orig_dataset_info.update(sft_dataset_info)
    orig_dataset_info.update(dpo_dataset_info)
    json.dump(orig_dataset_info, open(os.path.join(os.environ["LLAMAFACTORY_DIR"], "data/dataset_info.json"), "w"), indent=4)
    
    
if __name__ == '__main__':
    fire.Fire({
        # stage 1: ask
        'generate_mcq_from_captions': generate_mcq_from_captions, 
        # stage 2: think
        'generate_simple_cot': generate_simple_cot,
        'collect_simple_cot': collect_simple_cot, 
        # stage 3: think harder
        'generate_extended_cot': generate_extended_cot, 
        'collect_extended_cot': collect_extended_cot, 
        # create dataset
        'create_sft_dpo_dataset': create_sft_dpo_dataset, 
        'update_llamafactory_dataset_info': update_llamafactory_dataset_info,
    })