# LM-Lexicon: Improving Definition Modeling via Harmonizing Semantic Experts

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#training">Training</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#citation">Citation</a>
</p>

## Overview

**LM-Lexicon** is a research framework for **Definition Modeling** — generating contextual word definitions using LLMs. It supports inference and training across multiple backends with comprehensive evaluation.

**Tasks**: Word Interpretation · Context Synthesis · Definition Synthesis · Word Sense Disambiguation

**Features**: Multi-LLM support (GPT, Claude, Gemini, Llama, Qwen) · Mixture of Experts · BLEU/ROUGE/BERTScore/MoverScore/MAUVE evaluation · DeepSpeed/FSDP/LoRA training · Few-shot ICL

## Installation

```bash
# Clone and setup
git clone https://github.com/your-username/LM-Lexicon.git
cd LM-Lexicon
conda create -n lmlexicon python=3.10 && conda activate lmlexicon

# Core dependencies
pip install -r inference/requirements.txt

# Training (optional)
pip install deepspeed peft wandb flash-attn liger-kernel

# Evaluation metrics (optional)
pip install sacrebleu bert-score mauve-text nltk
```

Requires Python 3.8+, PyTorch 2.0+, CUDA 11.8+ (GPU).

## Quick Start

```bash
# Local model inference
cd inference/
python main.py \
  --task word-interpretation \
  --model /path/to/llama-3-8b \
  --prompt_path prompts/word-interpretation.txt \
  --input_path dataset/3D-EX/test.jsonl \
  --shot_num 0 --max_tokens 64 \
  --run_local_model --evaluate

# API model inference
python main.py \
  --task word-interpretation \
  --model gpt-4-turbo \
  --api_key YOUR_API_KEY \
  --prompt_path prompts/word-interpretation.txt \
  --input_path dataset/3D-EX/test.jsonl \
  --shot_num 3 --evaluate
```

## Training

```bash
cd training/

# Single GPU
python finetune.py --model_config run_config/llama_config.json

# Multi-GPU with DeepSpeed
deepspeed --num_gpus=4 finetune.py \
  --model_config run_config/llama_config.json \
  --deepspeed run_config/deepspeed_config_zero2.json

# LoRA fine-tuning
python finetune.py --model_config run_config/llama_config.json --use_lora
```

<details>
<summary>Example config (run_config/llama_config.json)</summary>

```json
{
    "model_type": "llama-3",
    "model_name_or_path": "/path/to/llama-3-8b",
    "do_train": "True",
    "do_eval": "True",
    "do_predict": "True",
    "data_path_train": "dataset/3D-EX/train.jsonl",
    "data_path_valid": "dataset/3D-EX/valid.jsonl",
    "data_path_test": "dataset/3D-EX/test.jsonl",
    "output_dir": "trained_models/llama-3-8b-3dex",
    "batch_size": 32,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 4,
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "cutoff_len": 128
}
```
</details>

## Datasets

Supported: **3D-EX** (RANLP 2023) · **WordNet** · **Oxford** · **Wikipedia** · **Urban Dictionary**

Input format (JSONL):
```json
{"term": "frozen", "context": "frozen with horror", "definition": "unable to move or act because of fear", "source": "3D-EX"}
```

## Evaluation

**Lexical**: BLEU, ROUGE-L, METEOR, NIST · **Semantic**: BERTScore, MoverScore, MAUVE

We adopt the **one-to-many** evaluation protocol ([Huang et al., 2021](https://aclanthology.org/2021.acl-long.587/)), reporting the max score across all reference definitions for polysemous words.

## Project Structure

```
LMLexicon/
├── inference/          # Inference (API/local models, prompts, evaluation)
└── training/           # Training (MoE, DeepSpeed, LoRA, configs)
```

## References

1. Ishiwatari et al. (2019) - Learning to Describe Unknown Phrases with Local and Global Contexts
2. Huang et al. (2021) - Definition Modelling for Appropriate Specificity
3. Kong et al. (2022) - Multitasking Framework for Unsupervised Simple Definition Generation
4. Zhang et al. (2022) - Fine-grained Contrastive Learning for Definition Generation
5. Giulianelli et al. (2023) - Interpretable Word Sense Representations via Definition Generation

## Citation

```bibtex
@article{liu2026lm,
  title={LM-Lexicon: Improving Definition Modeling via Harmonizing Semantic Experts},
  author={Liu, Yang and Yang, Jiaye and Li, Weikang and Liang, Jiahui and Li, Yang and Yan, Lingyong},
  journal={arXiv preprint arXiv:2602.14060},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
