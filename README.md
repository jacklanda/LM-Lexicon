# LM-Lexicon: Improving Definition Modeling via Harmonizing Semantic Experts

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

## Overview

**LM-Lexicon** is a research framework for **Definition Modeling** — generating contextual word definitions using Large Language Models (LLMs). This project provides tools for both inference and training, supporting multiple LLM backends and comprehensive evaluation metrics.

### Key Tasks

| Task | Description |
|------|-------------|
| **Word Interpretation** | Generate definitions for words given their context |
| **Context Synthesis** | Generate contextual sentences containing a given word |
| **Definition Synthesis** | Generate diverse definitions based on reference definitions |
| **Word Sense Disambiguation** | Disambiguate word senses in context |

## Features

- 🤖 **Multi-LLM Support**: OpenAI GPT, Anthropic Claude, Google Gemini, and local models (Llama, Qwen)
- 🧩 **Mixture of Experts (MoE)**: Custom sparse model implementation for efficient inference
- 📊 **Comprehensive Evaluation**: BLEU, ROUGE, METEOR, BERTScore, MoverScore, MAUVE, and more
- 🚀 **Efficient Training**: DeepSpeed, FSDP, LoRA, Flash Attention 2, and Liger Kernel optimization
- 📝 **In-Context Learning**: Flexible few-shot prompting with customizable templates

## Project Structure

```
LMLexicon/
├── inference/                 # Inference module
│   ├── model/                # Model utilities (API/local model support)
│   ├── prompts/              # Prompt templates
│   ├── type/                 # Type definitions
│   ├── main.py               # Main inference script
│   ├── args.py               # Argument parser
│   ├── eval.py               # Evaluation metrics
│   └── requirements.txt      # Dependencies
│
├── training/                  # Training module
│   ├── moe/                  # Mixture of Experts implementation
│   │   ├── models/           # MoE model definitions
│   │   ├── composers/        # Expert composition utilities
│   │   └── compose_layers.py # MoE layer implementation
│   ├── eval/                 # Training evaluation tools
│   ├── utils/                # Training utilities
│   ├── run_config/           # Training configurations
│   └── finetune.py           # Main training script
│
└── LM-Lexicon/               # Documentation & references
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/LM-Lexicon.git
cd LM-Lexicon

# Create virtual environment
conda create -n lmlexicon python=3.10
conda activate lmlexicon

# Install dependencies
pip install -r inference/requirements.txt

# For training (additional dependencies)
pip install deepspeed peft wandb flash-attn liger-kernel
```

### Additional Dependencies

```bash
# Evaluation metrics
pip install sacrebleu bert-score mauve-text nltk

# API clients (if using cloud LLMs)
pip install openai anthropic google-generativeai
```

## Quick Start

### Inference with Local Model

```bash
cd inference/

python main.py \
  --task word-interpretation \
  --model /path/to/llama-3-8b \
  --prompt_path prompts/word-interpretation.txt \
  --input_path dataset/3D-EX/test.jsonl \
  --shot_num 0 \
  --max_tokens 64 \
  --run_local_model \
  --evaluate
```

### Inference with OpenAI API

```bash
python main.py \
  --task word-interpretation \
  --model gpt-4-turbo \
  --api_key YOUR_API_KEY \
  --prompt_path prompts/word-interpretation.txt \
  --input_path dataset/3D-EX/test.jsonl \
  --shot_num 3 \
  --evaluate
```

## Usage

### Inference

#### Command Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--task` | str | Task type: `word-interpretation`, `synthesize-context`, `synthesize-definition`, `word-sense-disambiguation` |
| `--model` | str | Model name/path (required) |
| `--prompt_path` | str | Path to prompt template (required) |
| `--input_path` | str | Path to input data (required) |
| `--shot_num` | int | Number of few-shot examples |
| `--api_key` | str | API key (for cloud LLMs) |
| `--base_url` | str | API base URL |
| `--run_local_model` | flag | Use local model instead of API |
| `--with_vllm` | flag | Use vLLM for deployment |
| `--max_query` | int | Maximum number of queries |
| `--max_tokens` | int | Maximum tokens to generate |
| `--temperature` | float | Sampling temperature |
| `--top_p` | float | Nucleus sampling threshold |
| `--evaluate` | flag | Enable evaluation |

#### Supported Models

**API Models:**
- OpenAI: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`
- Anthropic: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- Google: `gemini-pro`, `gemini-1.5-pro`

**Local Models:**
- Dense: `Llama-3-8B`, `Llama-3-70B`, `Qwen-2.5-7B`, `Qwen-2.5-72B`
- Sparse (MoE): `Llama-3-MoE`

### Training

#### Single GPU Training

```bash
cd training/

python finetune.py \
  --model_config run_config/llama_config.json
```

#### Multi-GPU Training with DeepSpeed

```bash
deepspeed --num_gpus=4 finetune.py \
  --model_config run_config/llama_config.json \
  --deepspeed run_config/deepspeed_config_zero2.json
```

#### LoRA Fine-tuning

```bash
python finetune.py \
  --model_config run_config/llama_config.json \
  --use_lora
```

#### Training Configuration

Create a JSON config file (e.g., `run_config/llama_config.json`):

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

## Datasets

### Supported Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| **3D-EX** | RANLP 2023 | Unified definitions and examples dataset |
| **WordNet** | Princeton | English lexical database |
| **Oxford** | Oxford Dictionary | English dictionary definitions |
| **Wikipedia** | Wikipedia | Encyclopedia-based definitions |
| **Urban** | Urban Dictionary | Slang and informal definitions |

### Data Format

Input data should be in JSONL format:

```json
{
    "term": "frozen",
    "context": "frozen with horror",
    "definition": "unable to move or act because of fear",
    "source": "3D-EX"
}
```

## Evaluation

### Evaluation Metrics

#### Lexical-Level Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **BLEU** | N-gram overlap | HuggingFace / SacreBLEU / NLTK / CPP |
| **ROUGE-L** | Longest common subsequence | HuggingFace |
| **METEOR** | Synonym-aware matching | NLTK |
| **NIST** | Information gain weighted | NLTK |

#### Semantic-Level Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| **BERTScore** | BERT embedding similarity | bert-score |
| **MoverScore** | Word embedding optimal transport | moverscore |
| **MAUVE** | Distribution divergence | mauve-text |

### Multi-Reference Evaluation

We adopt the **one-to-many** evaluation protocol following [Huang et al. (2021)](https://aclanthology.org/2021.acl-long.587/):
- Each word may have multiple reference definitions
- Report the maximum score across all references
- Provides fairer evaluation for polysemous words

### Running Evaluation

```bash
# Evaluate inference results
python main.py \
  --task word-interpretation \
  --model /path/to/model \
  --input_path dataset/3D-EX/test.jsonl \
  --evaluate \
  --run_local_model
```

## Technical Highlights

### Mixture of Experts (MoE)

Our custom MoE implementation supports:
- **Expert Composition**: Combine experts from multiple pretrained models
- **Top-k Routing**: Sparse activation for efficient inference
- **Layer-wise MoE**: Replace specific layers with MoE layers

```python
class MoeLayer(nn.Module):
    """
    Mixture of Expert Layer
    - gate: Router network
    - experts: List of expert networks
    - num_experts_per_tok: Number of experts activated per token
    """
```

### Training Optimizations

| Optimization | Benefit |
|--------------|---------|
| **Liger Kernel** | Fused operators for faster training |
| **Flash Attention 2** | Memory-efficient attention computation |
| **DeepSpeed ZeRO** | Distributed training with memory optimization |
| **Gradient Checkpointing** | Reduced memory footprint |
| **BF16/TF32** | Mixed precision training |

## Citation

If you find this work useful, please cite:

```bibtex
@article{lm-lexicon2024,
  title={LM-Lexicon: Improving Definition Modeling via Harmonizing Semantic Experts},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## References

Our evaluation methodology builds upon these works:

1. Ishiwatari et al. (2019) - Learning to Describe Unknown Phrases with Local and Global Contexts
2. Huang et al. (2021) - Definition Modelling for Appropriate Specificity
3. Kong et al. (2022) - Multitasking Framework for Unsupervised Simple Definition Generation
4. Zhang et al. (2022) - Fine-grained Contrastive Learning for Definition Generation
5. Giulianelli et al. (2023) - Interpretable Word Sense Representations via Definition Generation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [vLLM](https://github.com/vllm-project/vllm)
- [MoverScore](https://github.com/AIPHES/emnlp19-moverscore)

---

<p align="center">
  <i>For questions and issues, please open a GitHub issue or contact the authors.</i>
</p>
