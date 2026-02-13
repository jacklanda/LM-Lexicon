# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2024/01/30

import os
import torch
import argparse
import numpy as np
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import json
import openai
import requests
import anthropic
from httpx import Client

# from openai import AzureOpenAI
from rich.console import Console
from google import generativeai as genai
from anthropic import HUMAN_PROMPT, AI_PROMPT
from torch.nn.attention import SDPBackend, sdpa_kernel
from mergoo.models.modeling_llama import LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


HTTPX_CLIENT = Client(
    proxies={"https://": "http://127.0.0.1:7890", "http://": "http://127.0.0.1:7890"}
)

spacial_tokens_tensor = torch.tensor([128000, 128001, 128009]).to("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def query_local_model(
    args: argparse.Namespace,
    prompt: str,
    console: Console,
    local_model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    rerank: Optional[bool] = True,
    dedup: Optional[bool] = True,
    **kwargs,
) -> str:
    """
    Query local model.

    Args:
    - args: argparse.Namespace
    - prompt: str

    Returns:
    - response: str
    """
    generation_config = {
        "do_sample": True,
        "use_cache": True,
        # "max_length": 128,
        "temperature": args.temperature,
        # "top_k": 50,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        # "min_new_tokens": 4,
        "max_new_tokens": args.max_tokens,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "pad_token_id": 128009,
        "decoder_start_token_id": 128000,
        # "early_stopping": False,
        # "num_beams": 1,
        # "num_beam_groups": 4,
        # "diversity_penalty": 1.0,
        "output_scores": False,
        "output_logits": True,
        "output_hidden_states": False,
        "output_attentions": False,
        "return_dict_in_generate": True,
        "num_return_sequences": args.num_return_sequences,
    }

    prompt = tokenizer.bos_token + prompt + "\n\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=None,
        add_special_tokens=False,
    )
    inputs = {k: v.to(local_model.device) for k, v in inputs.items()}

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        outputs = local_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_legacy_cache=False,
            **generation_config,
        )
    # remove the input_ids from the output sequences
    outputs.sequences = outputs.sequences[:, inputs["input_ids"].shape[1] :]

    # obtain log probabilities for each token and total sequence log probability
    token_log_probs, sequence_log_probs = compute_sequence_log_likelihood(outputs)

    if rerank:
        # sort the sequences by log probability
        sorted_indices = sequence_log_probs.argsort(descending=True)
        outputs.sequences = outputs.sequences[sorted_indices]
        # token_log_probs = token_log_probs[sorted_indices].cpu().tolist()
        sequence_log_probs = sequence_log_probs[sorted_indices].cpu().tolist()
        # sequence_log_probs = list(set(sequence_log_probs))
        # sequence_log_probs = sorted(
        # sequence_log_probs[sorted_indices].cpu().tolist(), reverse=True
        # )
        # dedup sequence log probs & token log probs
        # console.log("token_log_probs:", token_log_probs)
        # console.log("sequence_log_probs:", sequence_log_probs)
        # convert sequences log probs to sequence probs
        # sequence_probs = torch.exp(sequence_log_probs)
        # console.log("sequence_probs:", sequence_probs)

    # decode the output sequences
    output_list = []
    for output in outputs.sequences:
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        output_list.append(decoded_output)

    preds = output_list
    # binding the sequence log probs to the output sequences
    preds = list(zip(preds, sequence_log_probs))

    if dedup:
        preds = list(set(preds))
        # sort by sequence log probs
        preds = sorted(preds, key=lambda x: x[1], reverse=True)
        # console.log("preds:", preds)

    return preds


def compute_sequence_log_likelihood(outputs: Any) -> Tuple[Any, Any]:
    """
    Calculate log conditional likelihood for generated sequences

    Args:
        outputs: model outputs containing sequences and logits

    Returns:
        token_log_probs: log conditional probability for each token position, shape: [batch_size, seq_len]
        sequence_log_probs: total log probability for each sequence, shape: [batch_size]
    """
    sequences = outputs.sequences  # [batch_size, seq_len]
    logits = outputs.logits  # list of tensors [batch_size, vocab_size]

    # Stack logits list into tensor [batch_size, seq_len, vocab_size]
    logits = torch.stack(logits, dim=1)

    # Calculate log softmax
    log_probs = torch.log_softmax(logits, dim=-1)

    # Get log probabilities for each generated token
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=sequences.unsqueeze(-1)
    ).squeeze(-1)

    # Calculate total log likelihood for the sequence
    sequence_log_probs = token_log_probs.sum(dim=-1)

    return token_log_probs, sequence_log_probs


def query_llm_general(args: argparse.Namespace, prompt: str, console: Console) -> str:
    url = args.base_url

    max_context_length = 16384 - 1024 - args.max_tokens
    prompt = " ".join(prompt.split(" ")[: int(max_context_length)])
    payload = json.dumps(
        {
            "model": args.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    )
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {args.api_key}",
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
    }

    try:
        response = requests.request(
            "POST", url, headers=headers, data=payload, timeout=30
        )
        response.raise_for_status()
        parsed_response = response.json()
        if "choices" in parsed_response and len(parsed_response["choices"]) > 0:
            content = parsed_response["choices"][0]["message"]["content"]
            print(content)
            return content
        else:
            print("No content in the response")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    sleep(8)


def query_openai(
    args: argparse.Namespace, prompt: str, tok: Any, console: Console, **kwargs
) -> str:
    """
    Query OpenAI API.

    Args:
    - args: argparse.Namespace
    - prompt: str

    Returns:
    - response: str
    """
    # setting up OpenAI API key
    openai.api_key = (
        args.api_key if args.api_key is not None else os.environ.get("OPENAI_KEY")
    )

    # setting up OpenAI API base url
    if args.base_url:
        openai.api_base = args.base_url

    # use proxy
    if args.proxy:
        openai.proxy = (
            {"http": args.proxy, "https": args.proxy}
            if args.proxy is not None
            else None
        )

    # truncate for each prompt up to the limitation of `max_tokens`
    model = args.model
    client = openai.OpenAI(
        api_key=args.api_key, base_url=args.base_url, http_client=None
    )
    # client = AzureOpenAI(
    # api_key=args.api_key,
    # api_version="2024-02-01",
    # azure_endpoint=args.base_url,
    # )
    accumulate_count = 0
    if "gpt-4" in model:
        max_context_length = 8192 - 1024 - args.max_tokens
    elif model == "gpt-3.5-turbo":
        max_context_length = 16384 - 1024 - args.max_tokens
    else:
        max_context_length = 8192
    if "encode" in dir(tok):
        input_ids = tok.encode(prompt)
    else:
        input_ids = tok(prompt, return_tensors="pt")["input_ids"]
    prompt = tok.decode(
        input_ids[: max_context_length - args.max_tokens],
    )
    messages = [
        {"role": "user", "content": prompt},
    ]
    while True:
        try:
            # deprecated API
            # response = openai.ChatCompletion.create(
            # model=model,
            # messages=messages,
            # max_tokens=args.max_tokens,
            # temperature=args.temperature,
            # top_p=args.top_p,
            # presence_penalty=args.presence_penalty,
            # frequency_penalty=args.frequency_penalty,
            # logprobs=True,
            # top_logprobs=5,
            # **kwargs,
            # )
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                extra_body={  # supporting extra decoding params of vLLM
                    "use_beam_search": args.use_beam_search,
                    "early_stopping": args.early_stopping,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                    "best_of": args.num_return_sequences,
                }
                if args.with_vllm
                else None,
            )
        except Exception as e:
            if accumulate_count > 2:
                console.log("Retrying failed, bypass this example.")
                return ""
            accumulate_count += 1
            max_context_length = max_context_length - 512
            prompt = tok.decode(
                input_ids[: max_context_length - args.max_tokens],
            )  # decreasing prompt length
            messages = [
                {"role": "assistant", "content": prompt},
            ]
            console.log("Retrying:", str(e))
            sleep(3)  # consider enabling batch inference
        else:
            break

    if args.debug:
        top_two_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        for i, logprob in enumerate(top_two_logprobs, start=1):
            pred_detail = f"token: {logprob.token}\tlogprob: {logprob.logprob}\tprob: {np.exp(logprob.logprob) * 100:.2f}%"
            console.print(pred_detail)

    return (
        response.strip()
        if isinstance(response, str)
        else response.choices[0].message.content.strip()
    )


def query_claude(args: argparse.Namespace, prompt: str, console: Console) -> str:
    """
    Query Claude API.

    Args:
    - args: argparse.Namespace
    - prompt: str

    Returns:
    - response: str
    """
    accumulate_count = 0
    max_context_length = 16384 - 1024 - args.max_tokens
    prompt = " ".join(prompt.split(" ")[: int(max_context_length)])
    client = anthropic.Anthropic(  # anthropic.Client(
        api_key=args.api_key,
        timeout=1800,
        base_url=args.base_url,
        # http_client=HTTPX_CLIENT,
    )
    while True:
        try:
            if "claude-3" in args.model:
                response = (
                    client.messages.create(
                        model=args.model,
                        max_tokens=args.max_tokens,
                        system="",
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,  # [
                                # {
                                # "type": "text",
                                # "text": prompt,
                            }
                        ],
                    )
                    # .content[0]
                    # .text.strip()
                )
                # print("MESSAGE:", response)
                if response:
                    response = response.choices[0]["message"][
                        "content"
                    ]  # response.content[0].text.strip()
                    # print("CONTENT:", response)
                else:
                    response = ""
            else:
                response = client.completions.create(
                    model=args.model,
                    max_tokens_to_sample=args.max_tokens,
                    prompt=f"{HUMAN_PROMPT}{prompt}{AI_PROMPT}",
                )  # .completion.strip()
                # print("COMPLETION:", response)
                if response and response.completion:
                    response = response.completion.strip()
                else:
                    response = ""
        except Exception as e:
            if accumulate_count > 2:
                console.log("Retrying failed, bypass this instance.")
                return ""
            accumulate_count += 1
            prompt = " ".join(
                prompt.split(" ")[: (int(max_context_length - 512))]
            )  # decreasing prompt length
            console.log(f"Retrying: {type(e)}, {e}")
            sleep(3)
        else:
            if any(
                [p in response for p in ["Sorry", "Okay", "Unfortunately", "I don't"]]
            ):
                if accumulate_count > 2:
                    console.log("Retrying failed, return empty literal string.")
                    return ""
                console.log("Detecting rejective response, retry this instance.")
                accumulate_count += 1
                continue
            # print("final res:" + response)
            return response


def query_gemini(
    args: argparse.Namespace,
    prompt: str,
    console: Console,
    ensure_safety_response: bool = True,
) -> str:
    """
    Query Gemini API.

    Args:
    - args: argparse.Namespace
    - prompt: str
    - ensure_safety_response: bool

    Returns:
    - response: str
    """
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel(args.model)
    max_context_length = 30000 - args.max_tokens
    prompt = " ".join(prompt.split(" ")[: int(max_context_length)])
    safety_settings: Optional[List[Dict[str, str]]] = (None,)
    accumulate_count = 0
    if ensure_safety_response:
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
    while True:
        try:
            responses = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
                safety_settings=safety_settings,
            ).parts
            if len(responses) > 0:
                return responses[0].text.strip()
            else:
                if accumulate_count > 2:
                    return ""
                sleep(3)
                accumulate_count += 1
                continue
        except Exception as e:
            if accumulate_count > 2:
                console.log("Retrying failed, bypass this example.")
                return ""
            accumulate_count += 1
            prompt = " ".join(prompt.split(" ")[: (int(max_context_length - 512))])
            console.log(f"Retrying: {e}")
            sleep(3)


def get_gpu_memory_usage() -> List[Dict[str, Any]]:
    """
    Get GPU memory usage for all available GPUs using PyTorch
    Returns a list of tuples containing (total memory, used memory, usage percentage)
    """
    gpu_memory = []

    if not torch.cuda.is_available():
        return []

    try:
        for i in range(torch.cuda.device_count()):
            total_memory = (
                torch.cuda.get_device_properties(i).total_memory / 1024**2
            )  # Convert to MB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2

            usage_percent = (memory_allocated / total_memory) * 100

            gpu_memory.append(
                {
                    "device": i,
                    "total": round(total_memory, 3),
                    "reserved": round(memory_reserved, 3),
                    "allocated": round(memory_allocated, 3),
                    "percentage": round(usage_percent, 3),
                }
            )
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return []

    return gpu_memory


def display_gpu_usage() -> None:
    """Display GPU memory usage using rich.console.log"""
    console = Console()

    gpu_data = get_gpu_memory_usage()

    if not gpu_data:
        console.log(
            "[red]No GPU detected or error occurred while getting GPU information[/red]"
        )
        return

    for gpu in gpu_data:
        console.log(
            f"[cyan]GPU {gpu['device']}[/cyan]: "
            f"[magenta]{round(gpu['total'])} MB total[/magenta], "
            f"[yellow]{round(gpu['reserved'])} MB reserved[/yellow], "
            f"[green]{round(gpu['allocated'])} MB allocated[/green] "
            f"([red]{round(gpu['percentage'])}% used[/red])"
        )


def init_local_model(
    model_name: str, console: Console, model_type: str = "dense"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Initialize local model and tokenizer.

    Args:
    - model_name: str

    Returns:
    - model: AutoModelForCausalLM
    - tokenizer: AutoTokenizer
    """
    with console.status(f"[bold orchid1]Initalizing local model: {model_name}\n") as _:
        if model_type == "dense":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        elif model_type == "sparse":
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)

        # model = model.to("cuda:1")
        display_gpu_usage()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
            max_length=128,
        )
        tokenizer.bos_token_id = 128000
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token_id = 128009

    return model, tokenizer
