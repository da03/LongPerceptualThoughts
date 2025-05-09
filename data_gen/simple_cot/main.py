import os
import re
import json
import torch
import pandas as pd

from pathlib import Path
from utils import infer_template, MultipleChoicesRandomizer, convert_sft_simple_cot_dataset, convert_sft_thought_expansion_dataset, extract_options_from_user_prompt, get_unique_id
from itertools import islice
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.environ["LLAMAFACTORY_DIR"], "src"))


import ipdb
MODEL_NAME = "qwen2.5_vl_instruct"
QWEN2_5_VL_INSTRUCT_PATH = os.environ["QWEN2_5_VL_INSTRUCT_PATH"]


def initialize_dataset(config):
    from transformers import Seq2SeqTrainingArguments
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer

    dataset_name = "long_perceptual_thoughts/sft_docci_all_mcqs"
    vllm_config = {
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "max_model_len": config["max_model_len"],
    }
    vllm_config_str = json.dumps(vllm_config).replace("\"", "\'")
    model_args, data_args, _, _ = get_infer_args(dict(
        image_resolution=config["image_resolution"],
        cutoff_len=config["cutoff_len"],
        # DO NOT change the following
        model_name_or_path=QWEN2_5_VL_INSTRUCT_PATH,
        dataset=dataset_name,
        vllm_config=f"\"{vllm_config_str}\"",
        dataset_dir=os.path.join(os.environ["LLAMAFACTORY_DIR"], "data"),
        template=infer_template(QWEN2_5_VL_INSTRUCT_PATH),
        preprocessing_num_workers=8,
        infer_dtype="half",
        trust_remote_code=True,
        # tokenized_path="outputs/tokenized_path/" + dataset_name
    ))
    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)
    return model_args, data_args, template_obj, dataset_module, tokenizer

    
def initialize_vllm(config, template_obj, tokenizer):
    from llamafactory.extras.misc import get_device_count
    from vllm import LLM, SamplingParams
    
    sampling_kwargs = {
        "n": config["n"],
        "max_tokens": config["max_tokens"],
        # DO NOT change the following
        "repetition_penalty": 1.05, 
        "temperature": 0.7,
        "top_p": 0.8, 
        "top_k": -1,
        "stop_token_ids": template_obj.get_stop_token_ids(tokenizer),
        "skip_special_tokens": False,
    }
    sampling_params = SamplingParams(**sampling_kwargs)
    
    # Initialize vllm engine
    engine_args = {
        "max_num_seqs": config["max_num_seqs"],
        "model": QWEN2_5_VL_INSTRUCT_PATH,
        "trust_remote_code": True,
        "dtype": "half",
        "tensor_parallel_size": get_device_count() or 1,
        "pipeline_parallel_size": 1, 
        "disable_log_stats": True,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 1, "video": 0}
    
    vllm_config = {
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "max_model_len": config["max_model_len"],
    }
    engine_args.update(vllm_config)
        
    print(f"Engine args: {engine_args}")
    llm = LLM(**engine_args)
    return llm, sampling_params
    

def yield_chunks(dataset, metadata_df, template_obj, tokenizer, image_resolution, chunk_size=50):
    from llamafactory.extras.constants import IGNORE_INDEX
    
    inputs, prompts, labels, metadata = [], [], [], []
    counter = 0
    for sample in tqdm(dataset, desc="Preparing data"):
        if sample["images"]:
            multi_modal_data = {
                "image": template_obj.mm_plugin._regularize_images(sample["images"], image_resolution=image_resolution)
            }
        else:
            multi_modal_data = None

        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
        prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
        labels.append(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=False)
        )
        metadata_dict = metadata_df.iloc[counter].to_dict()
        metadata.append({
            "metadata": {
                "mcq_unique_id": metadata_dict["mcq_unique_id"],},
        })
        counter += 1
        if chunk_size > 0 and len(inputs) >= chunk_size:
            yield inputs, prompts, labels, metadata
            inputs, prompts, labels, metadata = [], [], [], []

    if inputs:
        yield inputs, prompts, labels, metadata
        
    
def generate_simple_cot_chunk(start, end, config, df, dataset_module, llm, sampling_params, tokenizer, template_obj):
    
    intermediate_df_path = Path(f"outputs/simple_cot/intermediate_df/{MODEL_NAME}/{start}_{end}.jsonl")
    intermediate_df_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(config, open(intermediate_df_path.parent / "config.json", "w"))
    thinking_bos_token = tokenizer.encode("<think>")
    
    
    n_processed_examples = 0
    if intermediate_df_path.exists():
        n_processed_examples += len(pd.read_json(intermediate_df_path, lines=True))
    
    dataset_module = islice(dataset_module["train_dataset"], start + n_processed_examples, end)
    df = df.iloc[start+n_processed_examples:end]
    print(f"Generating CoT for examples {start+n_processed_examples} to {end}")
    
    for inputs, prompts, labels, metadata in yield_chunks(
        dataset_module, 
        df, 
        template_obj=template_obj, 
        tokenizer=tokenizer, 
        image_resolution=config["image_resolution"], 
        chunk_size=config["chunk_size"]
    ):
        
        # Force thinking
        for i, example in enumerate(inputs):
            inputs[i]["prompt_token_ids"] = example["prompt_token_ids"] + thinking_bos_token
        
        # Force thinking
        results = llm.generate(inputs, sampling_params)
        # Add <think> back to response
        preds = [["<think>" + o.text for o in result.outputs] for result in results]
        
        with open(intermediate_df_path, "a", encoding="utf-8") as f:
            decoded_inputs = [tokenizer.decode(i["prompt_token_ids"]) for i in inputs]
            for prompt, pred, label, meta in zip(prompts, preds, labels, metadata):
                f.write(json.dumps({"prompt": prompt, "predict": pred, "label": label, "metadata": meta}, ensure_ascii=False) + "\n")
            
        torch.cuda.empty_cache()
        

def generate_simple_cot(start=0, end=100):
    mcq_gen_output_path = Path('outputs/docci_mcq.csv')
    df = pd.read_csv(mcq_gen_output_path, dtype=str, keep_default_na=False)
    
    # Initialize dataset
    # Only the following config may be changed
    config = {
        "image_resolution": 512 * 512,
        "cutoff_len": 1024,
        # vllm
        "max_model_len": 16384,
        # sampling
        "n": 10,
        "max_tokens": 1024,
        "max_num_seqs": 32,
        # chunking
        "chunk_size": 500
    }
    
    # Register docci_mcq first
    model_args, data_args, template_obj, dataset_module, tokenizer = initialize_dataset(config)
    llm, sampling_params = initialize_vllm(config, template_obj, tokenizer)
    
    assert len(df) == len(dataset_module["train_dataset"]), f"Data provided in 'outputs/docci_mcq.csv' \
        does not match the dataset size registered as `long_perceptual_thoughts/sft_docci_all_mcqs`."
    
    # Generate CoT
    generate_simple_cot_chunk(start, end, config, df, dataset_module, llm, sampling_params, tokenizer, template_obj)
    

think_answer_pattern = re.compile(r'<think>\s*(.*?)\s*</think>.*?<answer>\s*(.*?)\s*</answer>', re.DOTALL)
def extract_thought_and_answer(example):

    option_in_prompt = extract_options_from_user_prompt(example["_simple_cot_prompt"], example["mcq_question"])
    parsed_choice_list = MultipleChoicesRandomizer.parse_choice_list(option_in_prompt)
    parsed_choice_dict = {c[0]: c[1].lower() for c in parsed_choice_list}
    gt_answer = example["mcq_answer"]
    gt_answer_option = [c[0] for c in parsed_choice_list if c[1] == gt_answer]
    assert len(gt_answer_option) == 1, ipdb.set_trace()
        
    gt_answer_option = gt_answer_option[0]
    
    res = []
    for r in example["_simple_cot_raw_response"]:
        re_res = think_answer_pattern.findall(r)
        if len(re_res) == 0:
            res.append([None, None, None])
        else:
            thought = re_res[0][0].strip()
            answer = re_res[0][1].strip()

            hit = False
            if answer.lower() in parsed_choice_dict.values():
                # Answer is shown in text
                hit = True
            else:
                # Answer is show in options
                if answer == gt_answer_option:
                    # Possible answer: A
                    hit = True
                elif answer in [f.format(gt_answer_option) for f in MultipleChoicesRandomizer.answer_formats]:
                    # Possible answer: (A), A.
                    hit = True
                elif answer in [f.format(gt_answer_option, gt_answer) for f in MultipleChoicesRandomizer.choice_formats]:
                    # Possible answer: (A) text, A. text
                    hit = True
                elif answer.split(" ")[0] in [f.format(gt_answer_option) for f in MultipleChoicesRandomizer.answer_formats]:
                    # Possible answer: (A) XXXXX, A. XXXXX
                    hit = True
                    
                if hit:
                    # Fix the answer format based on the format specified in the prompt
                    answer = MultipleChoicesRandomizer.reformat_answer_option(answer, parsed_choice_list, input_prompt=example["_simple_cot_prompt"])
                
            # if not hit:
            #     print(f"GT: {gt_answer_option}, Parsed: {answer}")
            res.append([thought, answer, hit])

    return res

    
def collect_simple_cot():
    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True)

    mcq_gen_output_path = Path('outputs/docci_mcq.csv')
    df = pd.read_csv(mcq_gen_output_path, dtype=str, keep_default_na=False)
    
    dataset_name = "long_perceptual_thoughts/sft_docci_all_mcqs"
    from llamafactory.data.parser import get_dataset_list
    json_filename = str(get_dataset_list([dataset_name], "/h/andrewliao/research/visual_reasoning_pomdp/LLaMA-Factory/data")[0])
    registered_df = pd.read_json(json_filename)
    registered_df["mcq_unique_id"] = registered_df["metadata"].apply(lambda x: x["mcq_unique_id"])
    registered_df.drop(columns=["metadata", "images"], inplace=True)
    registered_df.rename(columns={"messages": "mcq_messages"}, inplace=True)
    df = pd.merge(df, registered_df, on='mcq_unique_id')
    
    # Collect all intermediate files
    simple_cot_config = json.load(open(Path(f"outputs/simple_cot/intermediate_df/{MODEL_NAME}/config.json")))
    
    intermediate_df_paths = Path(f"outputs/simple_cot/intermediate_df/{MODEL_NAME}").glob("*.jsonl")
    intermediate_df_list = [pd.read_json(p, lines=True) for p in intermediate_df_paths]
    intermediate_df = pd.concat(intermediate_df_list, ignore_index=True)
    intermediate_df["mcq_unique_id"] = intermediate_df["metadata"].apply(lambda x: x["metadata"]["mcq_unique_id"])
    intermediate_df.drop(columns=["metadata", "label"], inplace=True)
    intermediate_df.rename(columns={
        "prompt": "_simple_cot_prompt",
        "predict": "_simple_cot_raw_response",
    }, inplace=True)
    
    merged_df = pd.merge(df, intermediate_df, on='mcq_unique_id')
    merged_df.drop_duplicates(subset=["_simple_cot_raw_response"], keep='first', inplace=True)
    
    merged_df["_thought,answer,hit"] = merged_df.parallel_apply(extract_thought_and_answer, axis=1)
    merged_df["simple_cot_parsed_thought"] = merged_df["_thought,answer,hit"].apply(lambda x: [y[0] for y in x])
    merged_df["simple_cot_parsed_answer"] = merged_df["_thought,answer,hit"].apply(lambda x: [y[1] for y in x])
    merged_df["simple_cot_parsed_correct"] = merged_df["_thought,answer,hit"].apply(lambda x: [y[2] for y in x])
    
    exploded_df = merged_df.explode(["_simple_cot_raw_response", "simple_cot_parsed_thought", "simple_cot_parsed_answer", "simple_cot_parsed_correct"])
    exploded_df = exploded_df[exploded_df.apply(lambda x: x["simple_cot_parsed_thought"] is not None, axis=1)]
    exploded_df.drop(columns=["_thought,answer,hit"], inplace=True)
    exploded_df["simple_cot_unique_id"] = exploded_df["_simple_cot_raw_response"].apply(get_unique_id)
    
    
    print(f"Collected {len(exploded_df)} examples.")
    print(f"Overall Accuracy: {exploded_df['simple_cot_parsed_correct'].mean():.3f}")
    print(f"#Unique Images: {len(exploded_df['image_id'].unique())}")
    print(f"#Unique MCQs: {len(exploded_df['mcq_unique_id'].unique())}")
    
    
    SAMPLE_RATIO = 1 / 3.75
    correct_df = exploded_df[exploded_df["simple_cot_parsed_correct"] == True]
    for tag in ["500_images", "1000_images", "2000_images", "4000_images"]:
        if Path(f"outputs/docci_{tag}.json").exists():
            image_list = json.load(open(f"outputs/docci_{tag}.json"))
            convert_sft_simple_cot_dataset(correct_df, image_list, f"outputs/sft_docci_{tag}_simple_cots_weighted_sample.json", weighted_sample=True, sample_ratio=SAMPLE_RATIO)
            
    convert_sft_simple_cot_dataset(correct_df, None, f"outputs/sft_docci_all_simple_cots_weighted_sample.json", weighted_sample=True, sample_ratio=SAMPLE_RATIO)
    
    # Create for thought-expansion
    exploded_df.to_csv(Path(f"outputs/simple_cot/{MODEL_NAME}.csv"), index=False)
    convert_sft_thought_expansion_dataset(exploded_df, "outputs/sft_docci_thought_expansion.json")