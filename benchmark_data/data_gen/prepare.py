import io
import re
import base64
import json
import random
import ast
import os
from PIL import Image
from io import BytesIO
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from jinja2.sandbox import SandboxedEnvironment


def _generate_jsonl_file(df, system_prompt_template, dataset_name):
    data = []
    for i, row in df.iterrows():
        
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        
        # build prompt
        task_specific_message = f"Please provide your answer as (X), where X is the letter of the correct option."
        system_prompt = system_prompt_template.render(task_specific_message=task_specific_message)
            
        if "image_path" in row:
            user_prompt = f"<image>{question}"
        else:
            user_prompt = question
        
        user_prompt += " Select the correct answer from the following options:\n"
        user_prompt += "\n".join(choices)
        
        
        answer_prompt = f"<answer>{answer}</answer>"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer_prompt}
        ]
        if "image_path" in row:
            image_path = row["image_path"]
            images = [image_path]
            
            data.append({
                "messages": messages,
                "images": images,
                "index": row["index"]
            })
        else:
            data.append({
                "messages": messages,
                "index": row["index"]
            })
    return data
        
    
def create_dataset_info():
    
    def _generate_jsonl_file_given_system_prompt_path(prefix, dataset_name, df, system_prompt_template_path):
        system_prompt_template = env.from_string(open(system_prompt_template_path).read())
        data = _generate_jsonl_file(df, system_prompt_template, dataset_name)
        # save to json
        print(f"Dump {len(data)} samples to {dataset_name}.json")
        (Path("outputs") / f"benchmark_{prefix}{dataset_name}.json").write_text(json.dumps(data, indent=4))
        
    env = SandboxedEnvironment()
    
    for tsv_file in Path("outputs/tsv_files").glob("*.tsv"):
        dataset_name = tsv_file.stem
        
        df = pd.read_csv(tsv_file, sep="\t")
        df["choices"] = df["choices"].apply(ast.literal_eval)
        print(tsv_file)
        print(df.keys())
        
        prefix = "direct_answer_"
        system_prompt_template_path = "../data_gen/templates/direct_answer_system_prompt.jinja2"
        _generate_jsonl_file_given_system_prompt_path(prefix, dataset_name, df, system_prompt_template_path)
        
        prefix = ""
        system_prompt_template_path = "../data_gen/templates/think_system_prompt.jinja2"
        _generate_jsonl_file_given_system_prompt_path(prefix, dataset_name, df, system_prompt_template_path)
        
            
    # Prepare the dataset_info.json used in llama-factory
    dataset_info = {}
    for json_file in Path("outputs").glob("benchmark_*.json"):
        dataset_name = json_file.stem
        info = {
            "file_name": str(json_file.absolute()),
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
        if "images" in json.load(open(json_file))[0]:
            info["columns"]["images"] = "images"
            
        dataset_info.update({dataset_name: info})
    
    Path("benchmark_dataset_info.json").write_text(json.dumps(dataset_info, indent=4))
    
    # Merge it to `data/dataset_info.json`
    orig_dataset_info = json.load(open(os.path.join(os.environ["LLAMAFACTORY_DIR"], "data/dataset_info.json")))
    orig_dataset_info.update(dataset_info)
    json.dump(orig_dataset_info, open(os.path.join(os.environ["LLAMAFACTORY_DIR"], "data/dataset_info.json"), "w"), indent=4)
    
    
# Common columns: image_path, question, choices, answer
def prepare_bench():
    
    # V* Bench
    def _parse_v_star_choices(x):
        hint = "Answer with the option's letter from the given choices directly."
        choices = x["text"].replace(x["question"], "").replace(hint, "")
        choices = choices.strip().split("\n")
        return choices
    
    # TODO: Download vstar bench images to `./outputs`
    assert os.path.exists("./outputs/vstar_bench"), \
        "Please download vstar bench images to `./outputs/vstar_bench`" \
            "git clone https://huggingface.co/datasets/craigwu/vstar_bench" \
            "and run `python main.py prepare_bench` again."
        
    v_star_image_dir = Path("./outputs/vstar_bench")
    v_star_image_dir = str(v_star_image_dir.absolute())
    dataset = load_dataset("craigwu/vstar_bench")
    
    df = dataset["test"].to_pandas()
    df["image_path"] = df.apply(lambda x: os.path.join(v_star_image_dir, x["image"]), axis=1)
    df["question"] = df["text"].apply(lambda x: x.split("\n(A)")[0])
    df["choices"] = df.apply(_parse_v_star_choices, axis=1)
    df["answer"] = df["label"].apply(lambda x: f"({x})")
    df.drop(columns=["text", "label", "image"], inplace=True)
    df["index"] = list(range(len(df)))
    df.to_csv(os.path.join(os.environ["BENCHMARK_DATASET_DIR"], "tsv_files", "v_star_bench.tsv"), sep="\t")
    