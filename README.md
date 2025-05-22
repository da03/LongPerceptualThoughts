# LongPerceptualThoughts

A framework for enriching visual reasoning with long chain-of-thoughts. We introduce a synthetic dataset that distills System-2-style reasoning into System-1 visual tasks, improving perceptual grounding and transfer to language tasks.

[**paper**](https://arxiv.org/abs/2504.15362) |
[**website**](https://andrewliao11.github.io/LongPerceptualThoughts/) |
[**dataset host on Huggingface**](https://huggingface.co/datasets/andrewliao11/LongPerceptualThought) |
[**X post**](https://x.com/andrewliao11/status/1917602672493973818)

![](./assets/overall_pipeline.gif)

## News
- ⭐ 2025/05/09: released code for data generation
- ⭐ 2025/04/21: released paper and dataset

## Prerequisite
1. CUDA==11.8
2. torch==2.5.1
3. transformers>=4.51.3
4. xformers==v0.0.27.post2

## 🔧 Usage: Synthesizing the Dataset

<details>
<summary>Conda env setup</summary>

Here is the line-by-line commands to install conda environment:
<pre><code>conda create -n long_perceptual_thoughts python=3.11 -y
conda install gcc=9 gxx=9 cmake -c conda-forge
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=11.8  -c pytorch -c nvidia
pip install git+https://github.com/huggingface/transformers@b1a2de075de86564f7e635f3b31a68b5f33e4cac --no-cache-dir
conda install -c conda-forge accelerate==0.34.0 peft==0.12.0 trl==0.9.6 -y
conda install -c conda-forge fire openai pandarallel -y 
pip install xformers==v0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118 --no-deps
pip install setuptools_scm tqdm pandas omegaconf datasets==3.1.0

cd vllm/
python use_existing_torch.py
pip install -e . --no-build-isolation -v
</code></pre>

Alternatively, you can install conda environment using the provided <code>.yml</code> file
<pre><code>conda create --name long_perceptual_thoughts --file environment.yml

cd vllm/
python use_existing_torch.py
pip install -e . --no-build-isolation -v
</code></pre>

</details>

Note: Both LLaMA-Factory and vllm are actively developed open-source projecets and the code might break when there are version mismatches.


### Data synthesis

We provide a three-stage data synthesis pipeline using image-caption datasets (e.g., [google/DOCCI](https://huggingface.co/datasets/google/docci)) to generate:

- Multiple-choice questions (MCQs)
- Short chain-of-thoughts (CoTs)
- Long CoTs
The output is a JSON format compatible with LLaMA-Factory.
For details, see the data generation README at [here](./data_gen/README.md)

```bash
# Prepare DOCCI or your own dataset first
# Check the website at https://google.github.io/docci/#downloads
cd data_gen/caption_datasets/docci
wget https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines
wget https://storage.googleapis.com/docci/data/docci_images.tar.gz
tar -xvf docci_images.tar.gz

cd ../../
bash run_3_stages_test.sh
```

## 📚 Citation

If you find this repository helpful, please cite:

```bibtex
@misc{liao2025longperceptualthoughtsdistillingsystem2reasoning,
      title={LongPerceptualThoughts: Distilling System-2 Reasoning for System-1 Perception}, 
      author={Yuan-Hong Liao and Sven Elflein and Liu He and Laura Leal-Taixé and Yejin Choi and Sanja Fidler and David Acuna},
      year={2025},
      eprint={2504.15362},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.15362}, 
}
```
