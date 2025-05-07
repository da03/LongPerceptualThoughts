import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from jinja2.sandbox import SandboxedEnvironment
from utils import OpenAICacheClient, get_unique_id, convert_sft_mcq_dataset, DOCCI, MultipleChoicesRandomizer

tqdm.pandas()
import ipdb


force_use_cache = True
        
def check_MCQ_valid(example):
    parsed_choice_list = MultipleChoicesRandomizer.parse_choice_list(example["mcq_choices"])
    gt_answer = example["mcq_answer"]
    gt_answer_option = [c[0] for c in parsed_choice_list if c[1] == gt_answer]
    return len(gt_answer_option) == 1

            
def generate_mcq_from_captions(max_examples=10000, verbose=False):
    env = SandboxedEnvironment()
    mcq_gen_prompt_template = env.from_string(open("mcq_gen/templates/caption_to_mcq.jinja2").read())
    parse_to_mcq_prompt_template = env.from_string(open("mcq_gen/templates/parse_to_mcq.jinja2").read())

    openai_client = {}
    gpt_4o_mini_model_id = 'gpt-4o-mini-2024-07-18'
    gpt_4o_model_id = 'gpt-4o-2024-08-06'
    for model_name, model_id in zip(['gpt-4o-mini', 'gpt-4o'], [gpt_4o_mini_model_id, gpt_4o_model_id]):
        CACHE_DIR = Path('outputs/mcq_gen/.cache') / model_id
        CACHE_DIR.mkdir(exist_ok=True, parents=True)
        openai_client[model_name] = OpenAICacheClient(model_id=model_id, cache_dir=CACHE_DIR, force_use_cache=force_use_cache, verbose=verbose)
        
        
    # Construct prompt
    def _construct_mcq_gen_prompt(example):
        prompt = mcq_gen_prompt_template.render(image_description=example["description"])
        return prompt
    
    def _construct_parse_to_mcq_prompt(example):
        prompt = parse_to_mcq_prompt_template.render(response=example["_mcq_gen_raw_response"])
        return prompt
    
    def parse_responses(example):
        res_list = re.findall(r'<question>\s*(.*?)\s*</question>.*?<choices>\s*(.*?)\s*</choices>.*?<answer>\s*(.*?)\s*</answer>.*?<type>(.*?)</type>', res, re.DOTALL)
        
        
    dataset = DOCCI('train', max_examples=max_examples)
    image_id_list = sorted(dataset.df["image_id"].unique())
    import random
    random.seed(42)
    random.shuffle(image_id_list)
    
    # Generate MCQs
    system_prompt = "You are a helpful assistant good at converting image descriptions to multi-choice visual questions."
    dataset.df["_mcq_gen_model_id"] = gpt_4o_model_id
    dataset.df["_mcq_gen_prompt"] = dataset.df.progress_apply(_construct_mcq_gen_prompt, axis=1)
    dataset.df["_mcq_gen_raw_response"] = dataset.df.progress_apply(lambda x: openai_client['gpt-4o'].query_openai(x["_mcq_gen_prompt"], system_prompt, temperature=0.7, n=1)[0], axis=1)
    
    # Parse to MCQs
    system_prompt = "You are a helpful assistant good at converting parsing text contain several multiple-choice questions into the specified format."
    dataset.df["_parse_to_mcq_model_id"] = gpt_4o_mini_model_id
    dataset.df["_parse_to_mcq_prompt"] = dataset.df.progress_apply(_construct_parse_to_mcq_prompt, axis=1)
    dataset.df["_parse_to_mcq_response"] = dataset.df.progress_apply(lambda x: openai_client['gpt-4o-mini'].query_openai(x["_parse_to_mcq_prompt"], system_prompt, temperature=0., n=1)[0], axis=1)
    dataset.df["_temp"] = dataset.df.progress_apply(lambda x: re.findall(r'<question>\s*(.*?)\s*</question>.*?<choices>\s*(.*?)\s*</choices>.*?<answer>\s*(.*?)\s*</answer>.*?<type>(.*?)</type>', x['_parse_to_mcq_response'], re.DOTALL), axis=1)
    print('#mcqs per image (#mcqs, count):', Counter(dataset.df["_temp"].apply(lambda x: len(x))).most_common())
    
    dataset.df = dataset.df.explode("_temp")
    
    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)
    dataset.df["mcq_question"] = dataset.df.parallel_apply(lambda x: x["_temp"][0].strip(), axis=1)
    dataset.df["mcq_choices"] = dataset.df.parallel_apply(lambda x: x["_temp"][1].strip(), axis=1)
    dataset.df["mcq_answer"] = dataset.df.parallel_apply(lambda x: x["_temp"][2].strip(), axis=1)
    dataset.df["mcq_question_type"] = dataset.df.parallel_apply(lambda x: x["_temp"][3].strip(), axis=1)
    dataset.df.drop(columns=["_temp"], inplace=True)
    
    print('Distribution of the question types:', Counter(dataset.df["mcq_question_type"]).most_common())
    
    dataset.df["mcq_unique_id"] = dataset.df.progress_apply(lambda x: get_unique_id(str((x["mcq_question"], x["mcq_choices"], x["mcq_answer"], x["mcq_question_type"])), image_path=x["image_path"]), axis=1)
    
    # Usually around 1 - 1.5 % of data is invalid
    dataset.df["mcq_is_valid"] = dataset.df.progress_apply(check_MCQ_valid, axis=1)
    dataset.df = dataset.df[dataset.df["mcq_is_valid"]]
    
    Path("outputs").mkdir(exist_ok=True)
    json.dump(image_id_list[:500], open('outputs/docci_500_images.json', 'w'))
    json.dump(image_id_list[:1000], open('outputs/docci_1000_images.json', 'w'))
    json.dump(image_id_list[:2000], open('outputs/docci_2000_images.json', 'w'))
    json.dump(image_id_list[:4000], open('outputs/docci_4000_images.json', 'w'))
    dataset.df.to_csv('outputs/docci_mcq.csv', index=False)
    
    
    # Create MCQs
    sampled_df = dataset.df[dataset.df.apply(lambda x: x["image_id"] in image_id_list[:500], axis=1)]
    convert_sft_mcq_dataset(sampled_df, "outputs/sft_docci_500_images_mcqs.json")
    convert_sft_mcq_dataset(sampled_df, "outputs/sft_direct_answer_docci_500_images_mcqs.json", direct_answer=True)
        
    sampled_df = dataset.df[dataset.df.apply(lambda x: x["image_id"] in image_id_list[:1000], axis=1)]
    convert_sft_mcq_dataset(sampled_df, "outputs/sft_docci_1000_images_mcqs.json")
    convert_sft_mcq_dataset(sampled_df, "outputs/sft_direct_answer_docci_1000_images_mcqs.json", direct_answer=True)
    
    sampled_df = dataset.df[dataset.df.apply(lambda x: x["image_id"] in image_id_list[:2000], axis=1)]
    convert_sft_mcq_dataset(sampled_df, "outputs/sft_docci_2000_images_mcqs.json")
    convert_sft_mcq_dataset(sampled_df, "outputs/sft_direct_answer_docci_2000_images_mcqs.json", direct_answer=True)
        
    sampled_df = dataset.df[dataset.df.apply(lambda x: x["image_id"] in image_id_list[:4000], axis=1)]
    convert_sft_mcq_dataset(sampled_df, "outputs/sft_docci_4000_images_mcqs.json")
    convert_sft_mcq_dataset(sampled_df, "outputs/sft_direct_answer_docci_4000_images_mcqs.json", direct_answer=True)

    convert_sft_mcq_dataset(dataset.df, "outputs/sft_docci_all_mcqs.json")
    convert_sft_mcq_dataset(dataset.df, "outputs/sft_direct_answer_docci_all_mcqs.json", direct_answer=True)
    
    
model_name_to_path = {
    "qwen2.5-vl-7b-instruct": "/scratch/ssd004/scratch/selflein/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-3b-instruct": "/h/andrewliao/large-scratch/pretrained_weights/Qwen2.5-VL-3B-Instruct", 
    "qwen2-vl-7b-instruct": "/model-weights/Qwen2-VL-7B-Instruct",
    "qwen2-vl-2b-instruct": "/h/andrewliao/large-scratch/pretrained_weights/Qwen2-VL-2B-Instruct"
}
def evaluate_MCQ_difficulty(dataset="long_perceptual_thoughts/sft_direct_answer_docci_all_mcqs"):
    
    import sys
    sys.path.append("../")
    from slurm_utils import create_slrm_job_directory, create_slurm_script, partition_to_max_time
    
    qos = "m3"
    for model in ["qwen2.5-vl-7b-instruct", "qwen2.5-vl-3b-instruct", "qwen2-vl-7b-instruct", "qwen2-vl-2b-instruct"]:
        
        now = datetime.now()
        work_dir = Path("outputs/slurm_jobs") / f"{now.year}-{now.month}-{now.day}" / f"{now.hour}-{now.minute}-{now.second}-{now.microsecond}"
        output_dir, script_dir, log_dir = create_slrm_job_directory(work_dir)
        job_run_name = f"evaluate_MCQ_difficulty_{model}"
        
        prediction_dir = Path("outputs/mcq_gen/difficulty") / model
        prediction_dir.mkdir(exist_ok=True, parents=True)
        
        command = f"python vllm_eval.py predict_and_eval --eval_dataset {dataset} --do_eval False --model_path {model_name_to_path[model]} --prediction_dir {prediction_dir.absolute()} --temperature 0.7 --top_p 0.8 --top_k -1 --repetition_penalty 1.05 --n_samples 5 --force_thinking False --use_tokenized_dataset True"
        
        open(script_dir / "run.sh", "w").write(f"""
cd ../
{command}
""")
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
        
        os.system(command)