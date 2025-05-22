#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color


N_TEST_IMGS=5

# Set the chunk size for each stage
STAGE_2_CHUNK_SIZE=1000
STAGE_3_CHUNK_SIZE=500

# Feel free to change this phrase to any other cognitive phrase you want to use. 
# This phrase corresponds to the cognitive cues mentioned in Section 2.3 of the paper.
COGNITIVE_PHRASE="Wait,"


export DISABLE_VERSION_CHECK=1
export PROJECT_ROOT="/PATH/TO/GITHUB/ROOT/"
-export QWEN2_5_VL_INSTRUCT_PATH="/PATH/TO/QWEN2.5-VL-INSTRUCT-7B"
-export R1_DISTILLED_QWEN_32_B="/PATH/TO/R1-DISTILLED-QWEN-32B"
export LLAMAFACTORY_DIR="${PROJECT_ROOT}/LLaMA-Factory"


# Stage 1: Generate MCQ
# NOTE: we implement cache when querying API, 
# so you can run this stage multiple times without re-generating the same images
python main.py generate_mcq_from_captions --max_examples $N_TEST_IMGS
python main.py update_llamafactory_dataset_info   # this will in-place update the dataset info
echo -e "${GREEN}✔ MCQ generation complete${NC}"

# Stage 2: Generate simple CoT
# Explicily specify the number of GPUs to use if needed
# export CUDA_VISIBLE_DEVICES=0,1,2,3
for ((i=0; i<N_TEST_IMGS; i+=STAGE_2_CHUNK_SIZE)); do
    end=$((i + STAGE_2_CHUNK_SIZE))
    python main.py generate_simple_cot --start $i --end $end
done
python main.py collect_simple_cot
python main.py update_llamafactory_dataset_info
echo -e "${GREEN}✔ Simple CoT generation complete${NC}"

# Stage 3: Generate long CoT
for ((i=0; i<N_TEST_IMGS; i+=STAGE_3_CHUNK_SIZE)); do
    end=$((i + STAGE_3_CHUNK_SIZE))
    python main.py generate_extended_cot "|${COGNITIVE_PHRASE}|" --start $i --end $end
done
python main.py collect_extended_cot
python main.py create_sft_dpo_dataset SFT
echo -e "${GREEN}✔ Extended CoT generation complete${NC}"


python main.py update_llamafactory_dataset_info

echo -e "${GREEN}✔ All stages complete${NC}"
echo -e "${GREEN}✔ Now, you could fine the generated dataset at outputs/ ${NC}"