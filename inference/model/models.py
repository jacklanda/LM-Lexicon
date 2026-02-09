# -*- coding: utf-8 -*-
#
# @Author: YangLiu <yangliu.real@gmail.com>
# @Date: 2023/12/27

from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

import torch
from pydantic import BaseModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer


AVAILABLE_MODELS = [
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",  # GPT-4 Turbo
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-3.5-turbo-1106",  # Updated GPT 3.5 Turbo
    "gpt-3.5-turbo",  # Currently points to gpt-3.5-turbo-0613.
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301",
    "text-davinci-003",
    "claude-instant",
    "claude-instant-1",
    "claude-2",
    "claude-2.0",
    "claude-2.1",
    "claude-3-sonnet",
    "claude-3-sonnet-20240229",
    "claude-3-opus",
    "claude-3-opus-20240229",
    "gemini-pro",
    "deepseek-chat",
    "lmsys/vicuna-7b-v1.5",  # open model
    "THUDM/chatglm3-6b",  # open model
    "mistralai/Mistral-7B-Instruct-v0.2",  # open model
    "./Mistral-7B-Instruct-v0.2",  # open model
    "NousResearch/Llama-2-7b-chat-hf",  # open model
    "4bit/Llama-2-7b-chat-hf",  # open model
    "01-ai/Yi-6B-Chat",  # open model
    "llama2-13b-hf",  # open model
    "meta-llama/Llama-2-7b-chat-hf",  # open model
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "vicuna-13b-v1.5",  # open model
    "chatglm-2",  # open model
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mixtral-8×7b-instruct-v0.1",  # open model
]


# implement a dataclass to define a list of available models
@dataclass
class Model(BaseModel):
    name: str
    context_size: int
    publish_date: datetime
    description: Optional[str] = None
    organization: Optional[str] = None


class LogLikelihoodEstimator:
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        model_name: Optional[str] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        console: Optional[Console] = Console(),
        rerank: Optional[bool] = True,
        descending: Optional[bool] = True,
        verbal: Optional[bool] = False,
    ):
        """
        Multi-sample log-likelihood estimator for LM-Lexicon

        Args:
            model (AutoModelForCausalLM): pre-trained model
            model_name (str): pre-trained model name
            rerank (bool): whether to rerank the samples
            descending (bool): whether to sort the samples in descending order
            verbal (bool): whether to print out the log
        """
        self.console = console

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        # Load model
        if model is not None:
            self.model = model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

        # rerank parameters
        self.rerank = rerank
        self.descending = descending

        # log parameters
        self.verbal = verbal

    def compute(self, samples: List[str]) -> List[float]:
        """
        Compute the log-likelihood of samples

        Args:
            samples (List[str]): list of samples to compute log-likelihood

        Returns:
            List[float]: list of log-likelihoods
        """
        # Create full texts for all candidates at once
        # full_texts = [context + candidate for candidate in candidates]

        # Tokenize samples as a batch
        inputs = self.tokenizer(
            samples,
            padding=True,  # Enable padding
            truncation=True,  # Enable truncation if needed
            max_length=self.model.config.max_position_embeddings,  # Max length of the text
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Ensure input_ids and attention_mask are of type LongTensor
        inputs["input_ids"] = inputs["input_ids"].long()
        inputs["attention_mask"] = inputs["attention_mask"].long()

        # Compute log-likelihood
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
            # Get the token probabilities
            token_probs = torch.nn.functional.log_softmax(
                logits, dim=-1
            )  # Shape: (batch_size, seq_len, vocab_size)
            self.console.log(token_probs.shape)
            # self.console.log(token_probs[0, :, :10])
            # get the log-probability of the actual tokens
            log_probs = token_probs.gather(
                dim=-1, index=inputs["input_ids"].unsqueeze(-1)
            ).squeeze(-1)
            self.console.log("input_ids:", inputs["input_ids"].shape)
            self.console.log("log probability:", log_probs.shape)
            exit()

            outputs = self.model(**inputs, labels=inputs["input_ids"])

            # Convert logits to log probabilities using log_softmax
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)

            # Get log probability of each actual token
            # gather: For each position, get the log prob of the actual token
            # that appeared
            actual_token_log_probs = log_probs.gather(
                dim=-1, index=inputs["input_ids"].unsqueeze(-1)
            ).squeeze(-1)

            # Apply attention mask to handle padding
            # masked_fill will replace log probs of padding tokens with 0 so
            # they don't affect the sum
            masked_log_probs = actual_token_log_probs * inputs["attention_mask"]

            # Sum log probabilities for each sequence (equivalent to
            # multiplying probabilities)
            sequence_log_probs = masked_log_probs.sum(dim=1)

        return sequence_log_probs.tolist()
