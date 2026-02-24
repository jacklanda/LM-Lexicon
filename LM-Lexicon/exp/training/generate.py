# -*- coding: utf-8 -*-
#
# @author: Yang Liu <yangliu.real@gmail.com>
# @date: 2024/04/03

import os
import json
import argparse
from pprint import pprint
from typing import List, Dict, Union, Any

import torch
import transformers
from tqdm import tqdm
from peft import PeftModel
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def get_model(model_type: str, base_model: str) -> Any:
    """Load model from HuggingFace model hub or specified path pointed to checkpoint."""
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    if device == "cuda":
        if model_type == "llama":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                # device_map="auto",
                load_in_8bit=False,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                # device_map="auto",
                load_in_8bit=False,
            ).to(device)
        if os.path.exists(args.lora_weights):
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        if model_type == "llama":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
                load_in_8bit=True,
            )
        if os.path.exists(args.lora_weights):
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        if model_type == "llama":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                # device_map="auto",
                load_in_8bit=False,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, low_cpu_mem_usage=True, load_in_8bit=False
            )
        if os.path.exists(args.lora_weights):
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
            )

    return model


def load_dev_data(dev_file_path: str) -> List[Dict[str, Any]]:
    """Load dev data from file."""
    dev_data = []
    with open(dev_file_path) as f:
        lines = f.readlines()
        for line in lines:
            dev_data.append(json.loads(line.strip()))

    return dev_data


def generate_text(
    dev_data: List[Dict[str, Any]],
    batch_size: int,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
) -> List[Dict[str, Any]]:
    """Generate text from model."""
    res = list()
    for i in tqdm(
        range(0, len(dev_data) // batch_size, batch_size),
        total=len(dev_data) // batch_size,
        unit="batch",
    ):
        print("-" * 100)
        batch = dev_data[i : i + batch_size]
        batch_text = []
        for item in batch:
            # input_text = "human: " + item['instruction'] + item['input'] + "\n\nAssistant: "
            input_text = item["instruction"] + "\n\n" + item["input"]
            batch_text.append(
                tokenizer.bos_token + input_text
                if tokenizer.bos_token != None
                else input_text
            )

        with torch.autocast("cuda"):
            features = tokenizer(
                batch_text,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = features["input_ids"].to("cuda")
            attention_mask = features["attention_mask"].to("cuda")

            output_texts = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=5,
                # num_beam_groups=5,
                # no_repeat_ngram_size=3,
                temperature=0.5,
                top_p=1.0,
                top_k=100,
                # diversity_penalty=1.0,
                repetition_penalty=2.0,
                length_penalty=0.5,
                do_sample=False,  # disable greedy sampling as default
                min_new_tokens=128,
                max_new_tokens=1024,
                early_stopping=True,
            )
        output_texts = tokenizer.batch_decode(
            output_texts.cpu().numpy().tolist(),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        for i in range(len(output_texts)):
            input_text = batch_text[i]
            input_text = input_text.replace(tokenizer.bos_token, "")
            predict_text = output_texts[i][len(input_text) :]
            # define format of output datapoint
            res.append(
                {
                    "id": batch[i]["id"],
                    "input": input_text,
                    "predict": predict_text,
                    "target": batch[i]["output"],
                }
            )
        print("input:", res[-1]["input"])
        print("*" * 50)
        print("predict:", res[-1]["predict"])
        print("*" * 50)
        print("target:", res[-1]["target"])
        print("*" * 50)
        print()
    print("-" * 100)
    print()

    return res


def main(args):
    """Main function."""
    dev_data = load_dev_data(args.dev_file)
    res = generate_text(dev_data, batch_size, tokenizer, model)
    with open(args.output_file, "a+", encoding="utf-8") as f:
        for sample in res:
            f.write(json.dumps(sample, ensure_ascii=False, indent=4) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="model type of pretrained language model",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="pretrained language model",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="max length of dataset"
    )
    parser.add_argument("--dev_batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--lora_weights", default="", type=str, help="use lora")
    parser.add_argument("--output_file", type=str, default="data_dir/predictions.json")

    args = parser.parse_args()
    batch_size = args.dev_batch_size

    if args.model_type == "llama":
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.padding_side = "left"

    if args.model_type == "opt":
        tokenizer.padding_side = "left"
        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens({"bos_token": "<s>"})
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = get_model(args.model_type, args.model_name_or_path)

    main(args)
