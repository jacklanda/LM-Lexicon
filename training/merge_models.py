#!/usr/bin/env python
"""
Replaces ff layers using MOE. rest all will be averaged
"""

import torch
from rich.console import Console
from transformers import AutoTokenizer

from moe.compose_experts import ComposeExperts
from moe.models.modeling_llama import LlamaForCausalLM


console = Console()

# model_merge_type = "legacy"
model_merge_type = "3D-EX"

if model_merge_type == "3D-EX":
    model_id = "/mnt/buffer/liuyang/Meta-Llama-3-4x8B-MoE-3D-EX"
    config = {
        "model_type": "llama",
        "num_experts_per_tok": 2,
        "experts": [
            {
                "expert_name": "expert_0",
                "model_id": "/mnt/buffer/liuyang/Meta-Llama-3-8B-3D-EX-cluster-0",
            },
            {
                "expert_name": "expert_1",
                "model_id": "/mnt/buffer/liuyang/Meta-Llama-3-8B-3D-EX-cluster-1",
            },
            {
                "expert_name": "expert_2",
                "model_id": "/mnt/buffer/liuyang/Meta-Llama-3-8B-3D-EX-cluster-2",
            },
            {
                "expert_name": "expert_3",
                "model_id": "/mnt/buffer/liuyang/Meta-Llama-3-8B-3D-EX-cluster-3",
            },
        ],
        "router_layers": ["gate_proj", "up_proj", "down_proj"],
    }
elif model_merge_type == "legacy":
    model_id = "/mnt/buffer/liuyang/Meta-Llama-3-4x8B-MoE-Legacy"
    config = {
        "model_type": "llama",
        "num_experts_per_tok": 2,
        "experts": [
            {
                "expert_name": "expert_0",
                "model_id": "/mnt/buffer/liuyang/lm-lexicon-dense-wordnet-best-ckpt",
            },
            {
                "expert_name": "expert_1",
                "model_id": "/mnt/buffer/liuyang/lm-lexicon-dense-oxford-best-ckpt",
            },
            {
                "expert_name": "expert_2",
                "model_id": "/mnt/buffer/liuyang/lm-lexicon-dense-wiki-best-ckpt",
            },
            {
                "expert_name": "expert_3",
                "model_id": "/mnt/buffer/liuyang/lm-lexicon-dense-slang-best-ckpt",
            },
        ],
        "router_layers": ["gate_proj", "up_proj", "down_proj"],
    }

console.log("model merging started")
console.print_json(data=config, indent=4)

# create checkpoint
expertmerger = ComposeExperts(config, torch_dtype=torch.bfloat16)
expertmerger.compose()
expertmerger.save_checkpoint(model_id)

# load the merged checkkpoint
model = LlamaForCausalLM.from_pretrained(
    model_id
)  # 'gate' / router layers are untrained hence loaded warning would appeare for them
tokenizer = AutoTokenizer.from_pretrained(model_id)

console.log(
    "Input:",
    '<|begin_of_text|>"mix one volume of the solution with ten volumes of water" What is the definition of "volume"?\n\n',
)
outputs = model(
    torch.tensor(
        [
            [
                128000,
                1,
                36171,
                832,
                8286,
                315,
                279,
                6425,
                449,
                5899,
                27378,
                315,
                3090,
                1,
                3639,
                374,
                279,
                7419,
                315,
                330,
                26116,
                94770,
            ]
        ]
    )
)
console.print(outputs)

logits = outputs.logits

# get the most likely token index for each
predicted_ids = []
for i in range(len(logits[0])):
    predicted_index = torch.argmax(logits[0, i, :]).item()
    predicted_ids.append(predicted_index)
# get the predicted token
predicted_token = tokenizer.decode(predicted_ids)

# generated_tokens = tokenizer.decode(**generated_token_ids[0])
console.log("Output:", predicted_token)
console.log("model merging done")
