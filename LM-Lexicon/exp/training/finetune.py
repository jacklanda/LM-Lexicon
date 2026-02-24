# -*- coding: utf-8 -*-
#
# @author: Yang Liu <yangliu.real@gmail.com>
# @date: 2024/04/03
#
# LM-Lexicon: Towards Robust Lexical Definition Modeling

import os
import re
import sys
import json
import glob
import math
import copy
import logging
import argparse
import warnings
from statistics import mean
from ast import literal_eval
from typing import Any, List, Dict, Optional, Tuple, Union

import torch
import wandb
import numpy as np
import transformers
import editdistance
from paddlenlp.metrics import Distinct

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from evaluate import load
from datasets import load_dataset
from nltk.tokenize import word_tokenize

# from sentence_transformers import SentenceTransformer, util
from moe.models.modeling_llama import LlamaForCausalLM as MoELlamaForCausalLM
from emo_patch import (
    replace_llama_forward_with_emo_1_adaptive_forward,
    replace_llama_forward_with_emo_2_fixed_forward,
)
from peft import (
    LoraConfig,
    AutoPeftModelForCausalLM,
    prepare_model_for_kbit_training,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from eval import (
    compute_sentence_bleu,
    compute_nist,
    compute_rouge,
    compute_meteor,
    compute_mover,
    compute_bert_score,
    compute_exact_match,
    compute_sari,
    get_word2refs,
)
from utils import (
    IGNORE_INDEX,
    EarlyStoppingCallback,
    TrackGPUUtilizationCallback,
    EvaluateFirstStepCallback,
    TrackEvalResultCallback,
    timeit,
    dump_hf_dataset,
    get_lr_scheduler,
    get_adam_optimizer,
    get_adam_8bit_optimizer,
    unfreeze_all_layers,
    aggregate_definitions,
    optimize_specific_layers,
    create_sft_trainer,
    create_hpo_trainer,
    create_seq2seq_trainer,
    create_wandb_hp_space,
)

# set up wandb
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLED"] = "false"
os.environ["WANDB_ENTITY"] = "lm-lexicon"
os.environ["WANDB_PROJECT"] = "LM-Lexicon"

# set up logging
warnings.filterwarnings("ignore")
# transformers.logging.set_verbosity_info()

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

# set up scorers of evaluation metrics
BLEU_SCORER = load("bleu")
SACREBLEU_SCORER = load("sacrebleu")
GOOGLE_BLEU_SCORER = load("google_bleu")
# SENTENCE_ROUGE_SCORER = compute_rouge
ROUGE_SCORER = load("rouge")
# BERT_SCORER = load("bertscore")
# METEOR_SCORER = load("meteor")
# BLEURT_SCORER = load("bleurt")
# COSINE_SIMILARITY = SentenceTransformer("all-MiniLM-L6-v2")
# MAUVE_SCORER = load("mauve")
EDIT_DISTANCE_SCORER = lambda preds, golds, ignore_case: mean(
    [
        (
            editdistance.eval(p, g) / max(len(p), len(g))
            if not ignore_case
            else editdistance.eval(p.lower(), g.lower()) / max(len(p), len(g))
        )
        for (p, g) in zip(preds, golds)
    ]
)
DISTINCT_1_SCORER = Distinct(n_size=1)
DISTINCT_2_SCORER = Distinct(n_size=2)
DISTINCT_3_SCORER = Distinct(n_size=3)


def preprocess_logits_for_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> List[str]:
    """
    Original Trainer may cause OOM issue.
    This is a workaround to avoid storing too many tensors that are not needed.

    Args
    - logits(torch.Tensor): model output logits (batch_size, seq_len, vocab_size)
    - labels(torch.Tensor): target labels (batch_size, seq_len)

    Returns
    - pred_ids(torch.Tensor): predicted ids (batch_size, seq_len)
    - labels(torch.Tensor): target labels (batch_size, seq_len)
    """
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]

    # if pred_ids has 3 dimensions, reshape it to 2 dimensions with shape (batch_size, seq_len)
    if logits.dim() == 3:
        pred_ids = torch.argmax(logits, dim=-1)
    else:
        pred_ids = logits

    return pred_ids, labels


def get_logger(logger_name: str, output_dir: str) -> logging.Logger:
    """Initialize logger.

    Args:
    - logger_name(str): logger name
    - output_dir(str): output directory

    Returns:
    - logger(logging.Logger): logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "log.txt"), mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    return logger


def train(args: argparse.Namespace) -> None:
    """Training entry for supervised fine-tuning.

    Args:
    - args(argparse.Namespace): input arguments

    Returns:
    - None
    """
    model_config = json.load(open(args.model_config_file, "r"))
    do_train = True if literal_eval(model_config["do_train"]) else False
    do_eval = True if literal_eval(model_config["do_eval"]) else False
    do_predict = True if literal_eval(model_config["do_predict"]) else False
    model_type = model_config["model_type"]
    model_name_or_path = model_config["model_name_or_path"]
    train_mode = model_config["train_mode"]
    mask_term = True if model_config["mask_term"] == "True" else False
    data_path_train = model_config["data_path_train"]
    data_path_valid = model_config["data_path_valid"]
    data_path_test = model_config["data_path_test"]
    learning_rate = model_config["learning_rate"]
    eval_sample_ratio = model_config["eval_sample_ratio"]
    output_dir = model_config["output_dir"]
    max_seq_len = model_config["cutoff_len"]
    train_on_input = True if literal_eval(model_config["train_on_input"]) else False
    seq2seq_training = True if literal_eval(model_config["seq2seq_training"]) else False
    trainable_layers = model_config["trainable_layers"].split(",")
    fsdp = model_config["fsdp"]
    fsdp_wrap_layer = model_config["fsdp_wrap_layer"]
    run_name = model_config["run_name"]
    warmup_ratio = model_config.get("warmup_rate", 0.1)
    use_lora = args.use_lora

    # remove the output evaluation directory if exists
    if os.path.exists(f"{output_dir}/eval"):
        os.system(f"rm -rf {output_dir}/eval")
    os.makedirs(f"{output_dir}/eval", exist_ok=True)

    logger = get_logger("train", model_config["output_dir"])
    logger.info("args.__dict__ : {}".format(args.__dict__))
    for key, value in model_config.items():
        logger.info("{} : {}".format(key, value))

    assert model_name_or_path, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    gradient_accumulation_steps = (
        model_config["batch_size"] // model_config["per_device_train_batch_size"]
        if "gradient_accumulation_steps" not in model_config
        else model_config["gradient_accumulation_steps"]
    )

    logger.info(
        "per_device_train_batch_size = {}, gradient_accumulation_steps = {}".format(
            model_config["per_device_train_batch_size"], gradient_accumulation_steps
        )
    )
    device_map = "auto"  # set "auto" for `device_map` as default
    world_size = int(
        os.environ.get("WORLD_SIZE", 1)
    )  # `world_size` is corresponding to the number of GPUs
    ddp = world_size != 1
    if args.deepspeed and ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)

    # load model and tokenizer for LLaMA and its variants
    # EMO-1-Adaptive
    if model_type.lower() == "llama-3":
        # replace_llama_forward_with_emo_1_adaptive_forward(logger, model_type="dense")
        # replace_llama_forward_with_emo_2_fixed_forward(logger, model_type="dense")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        # register cost embedding for EMOLoss
        # cost_embedding = copy.deepcopy(model.lm_head.weight.data)
        # model.register_buffer("cost_embedding", cost_embedding)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
            # padding_side="right",
            # load_in_8bit=True if use_lora else False,
        )
        tokenizer.bos_token_id = 128000
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token_id = tokenizer.bos_token_id
    elif model_type.lower() == "llama-moe":
        # replace_llama_forward_with_emo_1_adaptive_forward(logger, model_type="sparse")
        logger.info("Using MoE-LLaMA model for training.")
        model = MoELlamaForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            moe_policy="ALL",
        )
        # frozen setting the parameters of the model
        if trainable_layers:
            model.enable_input_require_grads()
            model = optimize_specific_layers(model, trainable_layers)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
            # padding_side="right",
        )
        tokenizer.bos_token_id = 128000
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token_id = tokenizer.bos_token_id
    elif model_type.lower() == "llama-2":
        replace_llama_forward_with_emo_1_adaptive_forward(model_type="dense")
        model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
            # padding_side="right",
        )
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens({"bos_token": "<bos>"})
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "<eos>"})
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # load model and tokenizer for the last choice (default) of probing
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )

    # @timeit
    def compute_metrics(eval_pred: Tuple[Any], compute_result: bool = True):
        """Compute metrics for evaluation."""

        def postprocess_text(preds: List[str], labels: List[str]):
            """Preprocess text for computing metrics."""
            # heuristic rules for preprocessing
            preds = [
                " ".join(pred.replace("\n", "").split("?")[-1].split())
                for pred in preds
            ]
            preds = [p if (p != "" and not p.isspace()) else "<unk>" for p in preds]
            labels = [" ".join(label.replace("\n", "").split()) for label in labels]
            return preds, labels

        metrics = dict()
        if trainer.state.is_world_process_zero:

            if trainer.args.include_inputs_for_metrics:
                preds, labels, inputs = eval_pred
                inputs = np.where(inputs != IGNORE_INDEX, inputs, tokenizer.pad_token_id)
            else:
                preds, labels = eval_pred

            if trainer.args.predict_with_generate:
                if isinstance(preds, tuple):
                    preds = preds[0]
                if isinstance(labels, tuple):
                    labels = labels[0]
                if isinstance(inputs, tuple):
                    inputs = inputs[0]

                if not isinstance(preds, np.ndarray):
                    preds = preds.cpu().numpy()
                if not isinstance(labels, np.ndarray):
                    labels = labels.cpu().numpy()
                if not isinstance(inputs, np.ndarray):
                    inputs = inputs.cpu().numpy()

            # Replace IGNORE_INDEXs used for padding as we can't decode them
            # (batch_size * seq_len * hidden_dim)
            preds = np.where(preds != IGNORE_INDEX, preds, tokenizer.pad_token_id).astype(
                int
            )
            labels = np.where(
                labels != IGNORE_INDEX, labels, tokenizer.pad_token_id
            ).astype(int)

            # heuristic rules-based post-processing
            to_process_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            to_process_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            inputs_str = tokenizer.batch_decode(inputs, skip_special_tokens=True)
            preds_str, labels_str = postprocess_text(to_process_preds, to_process_labels)

            # extract word from each input samples
            words = [s.replace('"?', "").split('"')[-1].strip() for s in inputs_str]

            # construct word to multiple references mapping
            word2refs = get_word2refs(
                words=words,
                refs=labels_str,
                dedup_refs=False,
            )

            # add preds_str and labels_str to trainer
            trainer.state.eval_pred = (f"{output_dir}/eval", preds_str, labels_str)
            trainer.state.inputs = inputs_str
            trainer.state.outputs = preds_str
            trainer.state.labels = labels_str

            assert len(preds_str) == len(labels_str) == len(inputs_str) == len(words), (
                f"Length mismatch between `preds_str`, `labels_str`, `inputs_str`, and `words` "
                f"in evaluation: {len(preds_str)}, {len(labels_str)}, {len(inputs_str)}, {len(words)}"
            )

            # compute BLEU
            bleu = BLEU_SCORER.compute(
                predictions=preds_str,
                references=labels_str,
                tokenizer=lambda s: s.split(),
            )["bleu"]
            metrics["bleu"] = math.ceil(bleu * 10000) / 100

            # compute SacreBLEU
            sacrebleu = SACREBLEU_SCORER.compute(
                predictions=preds_str,
                references=labels_str,
            )["score"]
            metrics["sacrebleu"] = math.ceil(sacrebleu * 100) / 100

            # compute Google BLEU
            google_bleu = GOOGLE_BLEU_SCORER.compute(
                predictions=preds_str,
                references=labels_str,
            )["google_bleu"]
            metrics["google_bleu"] = math.ceil(google_bleu * 10000) / 100

            # compute sentence BLEU (nltk)
            sentence_bleu = compute_sentence_bleu(
                preds=preds_str,
                refs=labels_str,
                mode="nltk",
                words=words,
                word2refs=word2refs,
            )
            metrics["sentence_bleu"] = math.ceil(sentence_bleu * 10000) / 100

            # compute sentence BLEU (cpp)
            sentence_bleu_cpp = compute_sentence_bleu(
                preds=preds_str,
                refs=labels_str,
                mode="cpp",
                words=words,
                word2refs=word2refs,
            )
            metrics["sentence_bleu_cpp"] = math.ceil(sentence_bleu_cpp * 10000) / 100

            # compute ROUGE-L
            # rouge = ROUGE_SCORER.compute(
            # predictions=preds_str,
            # references=labels_str,
            # tokenizer=lambda s: s.split(),
            # )
            # metrics["rouge-1"] = math.ceil(rouge["rouge1"] * 10000) / 100
            # metrics["rouge-2"] = math.ceil(rouge["rouge2"] * 10000) / 100
            # metrics["rouge-l"] = math.ceil(rouge["rougeL"] * 10000) / 100

            # compute sentence ROUGE with multiple references
            multi_refs_rouge = compute_rouge(
                preds=preds_str,
                refs=labels_str,
                words=words,
                word2refs=word2refs,
                metric_key="rougeL",
            )
            metrics["rouge-l"] = math.ceil(multi_refs_rouge * 10000) / 100

            # compute sentence MOVER with multiple references
            sentence_mover = compute_mover(
                preds=preds_str,
                refs=labels_str,
                words=words,
                word2refs=word2refs,
            )
            metrics["mover-score"] = math.ceil(sentence_mover * 10000) / 100

            # compute METEOR
            multi_refs_meteor = compute_meteor(
                preds=preds_str,
                refs=labels_str,
                words=words,
                word2refs=word2refs,
            )
            metrics["meteor"] = math.ceil(multi_refs_meteor * 10000) / 100

            # compute embedding-based cosine similarity
            # NOTE: cosine similarity is not a good measure for DM (not sensitive)
            # embed_pred, embed_label = COSINE_SIMILARITY.encode(
            # preds_str,
            # convert_to_tensor=True
            # ), COSINE_SIMILARITY.encode(
            # labels_str,
            # convert_to_tensor=True
            # )
            # cosine_sim = util.pytorch_cos_sim(embed_pred, embed_label).mean().item()

            # FIXME: BLEURT is not available for now (OOM, need a single device)
            # bleurt = BLEURT_SCORER.compute(
            # predictions=preds_str,
            # references=labels_str,
            # )["scores"].mean()

            # compute BertScore
            bertscore = compute_bert_score(
                preds=preds_str,
                refs=labels_str,
                words=words,
                word2refs=word2refs,
                metric_key="all",
            )
            metrics["bertscore-p"] = math.ceil(bertscore["precision"] * 10000) / 100
            metrics["bertscore-r"] = math.ceil(bertscore["recall"] * 10000) / 100
            metrics["bertscore-f1"] = math.ceil(bertscore["f1"] * 10000) / 100

            # compute MAUVE
            # try:
            # mauve = MAUVE_SCORER.compute(
            # predictions=preds_str,
            # references=labels_str,
            # num_buckets=int(len(preds_str) * 0.01)
            # if len(preds_str) > 100
            # else "auto",
            # featurize_model_name="gpt2",  # gpt2, gpt2-base, gpt2-large, gpt2-xl, gpt2-xxl
            # max_text_length=256,
            # device_id=2,
            # seed=42,
            # verbose=False,
            # ).mauve
            # except ValueError as _:
            # mauve = 0.0
            # except RuntimeError as _:
            # mauve = 0.0
            # metrics["mauve"] = math.ceil(mauve * 10000) / 100

            # compute EXACT MATCH
            exact_match = compute_exact_match(
                preds=preds_str,
                refs=labels_str,
                words=words,
                word2refs=word2refs,
            )
            metrics["exact_match"] = math.ceil(exact_match * 10000) / 100

            # compute EDIT DISTANCE
            edit_distance = EDIT_DISTANCE_SCORER(preds_str, labels_str, ignore_case=True)
            metrics["edit_distance"] = round(edit_distance, 2)

            # compute SARI
            sari = compute_sari(
                sources=inputs_str,
                preds=preds_str,
                refs=labels_str,
                words=words,
                word2refs=word2refs,
            )
            metrics["sari"] = math.ceil(sari * 100) / 100

            # compute NIST
            nist = compute_nist(
                preds_str,
                labels_str,
                words=words,
                word2refs=word2refs,
            )
            metrics["nist"] = math.ceil(nist * 10000) / 100

            # compute DISTINCT-1, DISTINCT-2, and DISTINCT-3
            distinct_1, distinct_2, distinct_3 = 0.0, 0.0, 0.0
            distinct_1_sum, distinct_2_sum, distinct_3_sum = 0.0, 0.0, 0.0
            for pred in preds_str:
                pred_tokens = pred.split()
                DISTINCT_1_SCORER.add_inst(pred_tokens)
                DISTINCT_2_SCORER.add_inst(pred_tokens)
                DISTINCT_3_SCORER.add_inst(pred_tokens)
                distinct_1_sum += (
                    DISTINCT_1_SCORER.score() if DISTINCT_1_SCORER.count != 0.0 else 0.0
                )
                distinct_2_sum += (
                    DISTINCT_2_SCORER.score() if DISTINCT_2_SCORER.count != 0.0 else 0.0
                )
                distinct_3_sum += (
                    DISTINCT_3_SCORER.score() if DISTINCT_3_SCORER.count != 0.0 else 0.0
                )
                DISTINCT_1_SCORER.reset()
                DISTINCT_2_SCORER.reset()
                DISTINCT_3_SCORER.reset()
            metrics["distinct-1"] = (
                math.ceil(round(distinct_1_sum / len(preds_str), 4) * 10000) / 100
            )
            metrics["distinct-2"] = (
                math.ceil(round(distinct_2_sum / len(preds_str), 4) * 10000) / 100
            )
            metrics["distinct-3"] = (
                math.ceil(round(distinct_3_sum / len(preds_str), 4) * 10000) / 100
            )

            # compute generation length
            generation_length = [len(s.split()) for s in preds_str]
            metrics["generation_length"] = round(mean(generation_length), 2)

        return metrics

    def tokenize_full_sequence(
        input_text: str,
        target_text: str,
        term: Optional[str] = None,
        add_eos_token: bool = True,
        tokenize_train: bool = True,
    ) -> Dict[str, str]:
        def tokenize(s: str, add_special_tokens: bool = False):
            result = tokenizer(
                s,
                truncation=True,
                max_length=max_seq_len,
                # padding="max_length",
                return_tensors=None,
                return_attention_mask=True,
                add_special_tokens=add_special_tokens,
            )
            result["labels"] = result["input_ids"].copy()
            return result

        if tokenize_train:
            full_sequence = input_text + target_text
            tokenized_full_sequence = tokenize(full_sequence)
            if not train_on_input:
                user_prompt = input_text
                tokenized_user_prompt = tokenize(user_prompt)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                tokenized_full_sequence["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_sequence["labels"][
                    user_prompt_len:
                ]
        else:
            tokenized_input_sequence = tokenize(input_text)
            tokenized_target_sequence = tokenize(target_text)
            tokenized_full_sequence = tokenize(input_text + target_text)
            if not train_on_input:
                tokenized_full_sequence["input_ids"] = [tokenizer.pad_token_id] * len(
                    tokenized_target_sequence["input_ids"]
                ) + tokenized_input_sequence["input_ids"]

                tokenized_full_sequence["labels"] = tokenized_target_sequence[
                    "input_ids"
                ] + [-100] * len(tokenized_input_sequence["input_ids"])

                # tokenized_full_sequence["labels"] = [-100] * len(
                # tokenized_input_sequence["input_ids"]
                # ) + tokenized_target_sequence["input_ids"]

            # do not leak labels to model while predicting response
            tokenized_full_sequence["attention_mask"] = [0] * len(
                tokenized_target_sequence["input_ids"]
            ) + [1] * len(tokenized_input_sequence["input_ids"])

            # tokenized_full_sequence["attention_mask"] = [1] * len(
            # tokenized_input_sequence["input_ids"]
            # ) + [0] * len(tokenized_target_sequence["input_ids"])

        return tokenized_full_sequence

    def tokenize_old(
        input_text: str,
        target_text: str,
        term: Optional[str] = None,
        add_eos_token: bool = True,
        tokenize_train: bool = False,
    ) -> Dict[str, str]:
        """Tokenize prompt and convert it to input_ids, attention_mask and labels.

        Args:
        - input_text(str): input text
        - target_text(str): target text
        - term(str): term to mask
        - add_eos_token(bool): whether to add eos token

        Returns:
        - result(Dict[str, str]): tokenized result
        """

        result = dict()

        term_ids = None
        if mask_term and term is not None:
            term_ids = tokenizer(
                term,
                truncation=False,
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )["input_ids"]

        inputs = tokenizer(
            input_text,
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        targets = tokenizer(
            target_text,
            truncation=False,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        inputs_len = len(inputs["input_ids"])
        targets_len = len(targets["input_ids"])

        # (1) len of inputs + len of targets < max_seq_len
        if seq2seq_training and inputs_len < max_seq_len and targets_len < max_seq_len:
            result["input_ids"] = [tokenizer.pad_token_id] * (
                max_seq_len - inputs_len - targets_len
            ) + inputs["input_ids"]
            # result["input_ids"] = inputs["input_ids"] + [tokenizer.pad_token_id] * (
            # max_seq_len - inputs_len
            # )
        elif not seq2seq_training and inputs_len + targets_len < max_seq_len:
            if tokenize_train:
                result["input_ids"] = (
                    inputs["input_ids"]
                    + targets["input_ids"]
                    + [tokenizer.eos_token_id]
                    * (max_seq_len - inputs_len - targets_len)
                )
            else:
                result["input_ids"] = inputs["input_ids"] + [tokenizer.pad_token_id] * (
                    max_seq_len - inputs_len
                )
        # (2) len of inputs + len of targets >= max_seq_len, keep targets and shrink inputs
        elif inputs_len + targets_len >= max_seq_len:
            print(
                f"[DROP] `inputs_len` + `targets_len` should be less than {max_seq_len} in data point: {input_text}."
            )
            return {"input_ids": [], "attention_mask": [], "labels": []}
            # return {"input_ids": [], "labels": []}

        if inputs_len <= 8:
            # drop the data point if the length of input_text is less than 8
            print(
                f"[DROP] `inputs_len` should be greater than 30 in input of data point: {input_text}."
            )
            return {"input_ids": [], "attention_mask": [], "labels": []}
            # return {"input_ids": [], "labels": []}

        # Construct labels, ignore the loss computing for tokens of prompt by assigning them with IGNORE_INDEX
        if train_on_input:
            # (1) Train on the full sequence
            if not mask_term:
                result["labels"] = result["input_ids"].copy()
            elif mask_term and term_ids is not None:
                # mask the term in the sequence, following the two step:
                # (1) replace the training labels of the term with IGNORE_INDEX in the input sequence
                # (2) replace corresponding attention mask of the term with 0 in the input sequence
                # FIXME: detect how many times the term appears in the input sequence, so we can decide mask all or not
                result["labels"] = result["input_ids"].copy()
                term_begin_idx = result["input_ids"].index(term_ids[0])
                term_end_idx = term_begin_idx + len(term_ids) - 1
                for idx in range(term_begin_idx, term_end_idx + 1):
                    result["labels"][idx] = IGNORE_INDEX
                    # result["attention_mask"][idx] = 0
            else:
                raise ValueError("Please check the `mask_term` argument!")
        else:
            # (2) Train on the target sequence
            if seq2seq_training:
                result["labels"] = targets["input_ids"].copy() + [
                    tokenizer.eos_token_id
                ] * (max_seq_len - len(targets["input_ids"]))
                # result["labels"] = targets["input_ids"] + [tokenizer.pad_token_id] * (
                # max_seq_len - len(targets["input_ids"])
                # )
            else:
                if tokenize_train:
                    result["labels"] = (
                        [IGNORE_INDEX] * len(inputs["input_ids"])
                        + targets["input_ids"].copy()
                        + [tokenizer.eos_token_id]
                        * (
                            max_seq_len
                            - len(inputs["input_ids"])
                            - len(targets["input_ids"])
                        )
                    )
                else:
                    result["labels"] = targets["input_ids"].copy() + [
                        tokenizer.eos_token_id
                    ] * (max_seq_len - len(targets["input_ids"]))

            assert len(result["input_ids"]) == len(
                result["labels"]
            ), "Length mismatch between `input_ids`and `labels`!"

        if len(result["input_ids"]) != len(result["labels"]):
            # drop the data point if the length of input_ids and labels is mismatched
            print(
                f"[DROP] Length mismatch between `input_ids` and `labels` in {input_text}!"
            )
            return {"input_ids": [], "labels": []}

        return result

    def generate_and_tokenize_prompt(datapoint: Dict[str, Any]) -> Dict[str, str]:
        """Generate and construct prompt constrained by a fixed size of window.
        Dynamically generate input sequence and target sequence for each training example.

        Args:
        - datapoint(Dict[str, Any]): datapoint

        Returns:
        - tokenized result
        """
        term = datapoint["term"].strip()
        input_text = (
            tokenizer.bos_token + datapoint["instruction"].strip() + "\n\n"
        )  # no use prefix of prompt for fine-tuning
        # TODO: prepare non-instruction tuning data

        # Construct instruction data in Llama 3 style
        target_text = datapoint["definition"].strip() + tokenizer.eos_token

        # Check the length of input_text and target_text
        if len(word_tokenize(input_text + target_text)) <= max_seq_len:
            # tokenize and return the input_text and target_text
            tokenized_text = tokenize_full_sequence(
                input_text, target_text, term, tokenize_train=True
            )
            return tokenized_text
        else:
            # drop the data point if the length of input_text and target_text is greater than max_seq_len
            # print(
            # f"[DROP] Length of `input_text` ⨁ `target_text` should be less than {max_seq_len} in data point: {input_text}."
            # )
            return {"input_ids": [], "attention_mask": [], "labels": []}
            # return {"input_ids": [], "labels": []}

    def generate_and_tokenize_prompt_valid_test(
        datapoint: Dict[str, Any],
    ) -> Dict[str, str]:
        """Generate and construct prompt constrained by a fixed size of window.
        Dynamically generate input sequence and target sequence for each validation and test example.

        Args:
        - datapoint(Dict[str, Any]): datapoint

        Returns:
        - tokenized result
        """
        term = datapoint["term"].strip()
        input_text = tokenizer.bos_token + datapoint["instruction"].strip() + "\n\n"

        # Construct instruction data in Llama 3 style
        target_text = datapoint["definition"].strip() + tokenizer.eos_token

        # Check the length of input_text and target_text
        if len(word_tokenize(input_text + target_text)) <= max_seq_len:
            # tokenize and return the input_text and target_text
            tokenized_text = tokenize_full_sequence(
                input_text, target_text, term, tokenize_train=False
            )
            return tokenized_text
        else:
            # drop the data point if the length of input_text and target_text is greater than max_seq_len
            # print(
            # f"[DROP] Length of `input_text` ⨁ `target_text` should be less than {max_seq_len} in data point: {input_text}."
            # )
            return {"input_ids": [], "attention_mask": [], "labels": []}
            # return {"input_ids": [], "labels": []}

    if use_lora:
        model = prepare_model_for_kbit_training(model)
        # lora_hyperparams = json.load(open(args.lora_hyperparams_file))
        # for key, value in lora_hyperparams.items():
        # logger.info("{} : {}".format(key, value))
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.logging.info("Using LoRA for training.")
        print(lora_config)  # print the config for sanity check
        model = get_peft_model(model, lora_config)

    if train_on_input:
        print("Training on the full sequence.")
    else:
        print("Training on the target sequence.")

    if mask_term:
        print("Masking term in the full sequence.")

    if do_train:
        data_train = load_dataset("json", data_files=data_path_train)["train"]
        # tokenize datapoints for training set
        train_data = (
            data_train.shuffle(keep_in_memory=True, seed=42)
            .map(generate_and_tokenize_prompt, num_proc=32, keep_in_memory=True)
            .filter(
                lambda x: len(x["input_ids"]) > 0 or len(x["input_ids"]) >= max_seq_len,
                num_proc=32,
                keep_in_memory=True,
            )
        )
        # drop useless columns except for "input_ids", "attention_mask", and "labels"
        train_data = train_data.remove_columns(
            [
                col
                for col in train_data.column_names
                if col not in ["input_ids", "attention_mask", "labels"]
                # if col not in ["input_ids", "labels"]
            ]
        )
        # TODO: dump the training data to a file for debugging
        logger.info("Tokenizing training set success")

    if do_eval and os.path.isfile(data_path_valid):
        data_valid = load_dataset("json", data_files=data_path_valid)["train"]
        # tokenize datapoints for validation set
        val_data = (
            data_valid.shuffle(keep_in_memory=True, seed=42)
            .map(
                # generate_and_tokenize_prompt,
                generate_and_tokenize_prompt_valid_test,
                num_proc=32,
                keep_in_memory=True,
            )
            .filter(
                lambda x: len(x["input_ids"]) > 0 or len(x["input_ids"]) >= max_seq_len,
                num_proc=32,
                keep_in_memory=True,
            )
        )
        if eval_sample_ratio:
            sample_size = int(len(val_data) * eval_sample_ratio)
            val_data = val_data.select(range(sample_size), keep_in_memory=True)
            # dump validation data
            dump_hf_dataset(val_data, save_path=f"{output_dir}/eval/valid.json")
        # drop useless columns except for "input_ids", "attention_mask", and "labels"
        val_data = val_data.remove_columns(
            [
                col
                for col in val_data.column_names
                if col not in ["input_ids", "attention_mask", "labels"]
                # if col not in ["input_ids", "labels"]
            ]
        )
        # TODO: dump the validation data to a file for debugging
        logger.info("Tokenizing validation set success")
    else:
        val_data = None

    if do_predict and os.path.isfile(data_path_test):
        data_test = load_dataset("json", data_files=data_path_test)["train"]
        # tokenize datapoints for test set
        test_data = (
            data_test.shuffle(keep_in_memory=True, seed=42)
            .map(
                # generate_and_tokenize_prompt,
                generate_and_tokenize_prompt_valid_test,
                num_proc=32,
                keep_in_memory=True,
            )
            .filter(
                lambda x: len(x["input_ids"]) > 0 or len(x["input_ids"]) >= max_seq_len,
                num_proc=32,
                keep_in_memory=True,
            )
        )
        if eval_sample_ratio:
            sample_size = int(len(test_data) * eval_sample_ratio)
            # don't select test data
            # test_data = test_data.select(range(sample_size), keep_in_memory=True)
            # test_words = [example["term"] for example in test_data]
            # dump test data
            dump_hf_dataset(test_data, save_path=f"{output_dir}/eval/test.json")
        # drop useless columns except for "input_ids", "attention_mask", and "labels"
        test_data = test_data.remove_columns(
            [
                col
                for col in test_data.column_names
                if col not in ["input_ids", "attention_mask", "labels"]
                # if col not in ["input_ids", "labels"]
            ]
        )
        # TODO: dump the test data to a file for debugging
        logger.info("Tokenizing test set success")
    else:
        test_data = None

    train_nums = len(train_data)
    valid_nums = len(val_data) if val_data is not None else 0
    test_nums = len(test_data) if test_data is not None else 0

    num_gpus = torch.cuda.device_count()
    total_steps = (
        train_nums
        // (
            gradient_accumulation_steps
            * model_config["per_device_train_batch_size"]
            * num_gpus
        )
        + 1
    ) * model_config["num_epochs"]
    eval_interval_steps = save_interval_steps = (
        total_steps // model_config["eval_times"]
    )
    warmup_steps = int(total_steps * warmup_ratio)

    generation_config = GenerationConfig(
        # max_length=max_seq_len,
        max_new_tokens=64,
        early_stopping=False,
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=128000,
        use_cache=True,
    )
    training_args = transformers.Seq2SeqTrainingArguments(
        do_train=do_train,
        do_eval=do_eval,
        do_predict=do_predict,
        generation_config=generation_config,
        per_device_train_batch_size=model_config["per_device_train_batch_size"],
        per_device_eval_batch_size=model_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.01,
        num_train_epochs=model_config["num_epochs"],
        learning_rate=learning_rate,
        lr_scheduler_type="polynomial",  # linear  # TODO: write a custom scheduler with end lr
        lr_scheduler_kwargs={"lr_end": learning_rate * 0.5},
        seed=42,
        data_seed=42,
        # fp16=True,  # use fp16 (half precision) for training as default
        bf16=True,  # ad hoc for Llama-2, 3
        tf32=True,  # ad hoc for NVIDIA Ampere GPUs
        bf16_full_eval=True,  # use bf16 for full evaluation for Llama 3
        # enable gradient(activation) checkpointing as default to save GRAM
        gradient_checkpointing=True,
        remove_unused_columns=False,
        log_level="info",
        logging_first_step=True,
        logging_dir="logs/tensorboard",
        logging_steps=model_config["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=eval_interval_steps,
        save_steps=eval_interval_steps * 100000000,
        # disable_tqdm=True,
        disable_tqdm=False,
        save_safetensors=True,
        # eval_accumulation_steps=16,
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        output_dir=output_dir,
        overwrite_output_dir=True,  # ⚠️  Whether to overwrite the output directory
        save_total_limit=1,
        include_inputs_for_metrics=True,
        include_num_input_tokens_seen=True,
        batch_eval_metrics=False,
        predict_with_generate=True,  # ⚠️  Whether to use generate to calculate generative metrics (ROUGE, BLEU).
        metric_for_best_model="sentence_bleu_cpp",  # bleu, sacrebleu, google_bleu, sentence_bleu, sentence_bleu_cpp, etc.
        greater_is_better=True,
        load_best_model_at_end=True,
        # used to avoid GPU socket timeouts when performing slow operations in distributed runnings
        ddp_timeout=7200,  # 120 mins
        # ddp_backend="gloo",
        # ddp_find_unused_parameters=False if ddp else None,
        ddp_find_unused_parameters=None,
        deepspeed=(args.deepspeed if args.deepspeed and (not args.use_lora) else None),
        group_by_length=True,
        include_tokens_per_second=False,  # ⚠️  Whether or not to compute the number of tokens per second per device for training speed metrics.
        report_to=["wandb", "tensorboard"],
        # report_to=["tensorboard"],
        run_name=run_name,
        # "full_shard auto_wrap", "full_shard auto_wrap offload", etc.
        fsdp=fsdp,
        # LlamaDecoderLayer, OPTDecoderLayer, etc.
        fsdp_transformer_layer_cls_to_wrap=fsdp_wrap_layer,
    )
    # adam_optim = get_adam_optimizer(model, training_args)
    # adamw_optim = get_adamw_optimizer(model, training_args)
    adam_8bit_optim = get_adam_8bit_optimizer(model, training_args)
    # adam_optim = get_adam_optimizer(model, training_args)
    # adagrad_optim = get_adagrad_optimizer(model, training_args)
    # lr_scheduler = get_lr_scheduler(
    # adam_8bit_optim,
    # training_steps=total_steps,
    # warmup_ratio=warmup_ratio,
    # lr_begin=learning_rate,
    # lr_end=learning_rate * 0.5,
    # )
    data_collator = transformers.DataCollatorForSeq2Seq(
        model=model,
        tokenizer=tokenizer,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
        padding=True,
        return_tensors="pt",
    )
    if train_mode == "sft":
        trainer = create_sft_trainer(
            model=model,
            optimizer=adam_8bit_optim,
            tokenizer=tokenizer,
            training_args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics,
            callbacks=[
                # TrackGPUUtilizationCallback,
                EarlyStoppingCallback,
                EvaluateFirstStepCallback,
                TrackEvalResultCallback,
            ],
            data_collator=data_collator,
            device_map=device_map,
        )
    elif train_mode == "seq2seq":
        raise NotImplementedError("Seq2Seq training is not supported yet.")
    elif train_mode == "hpo":
        trainer = create_hpo_trainer(
            model=model,
            optimizer=adam_8bit_optim,
            tokenizer=tokenizer,
            training_args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            device_map=device_map,
        )
    else:
        raise ValueError(f"Invalid training mode: {train_mode}")

    logger.info("Training subprocess: {}".format(os.getpid()))

    if trainer.state.is_world_process_zero:
        logger.info(
            "bos token: {}\tbos token id: {}".format(
                tokenizer.bos_token, tokenizer.bos_token_id
            )
        )
        logger.info(
            "eos token: {}\teos token id: {}".format(
                tokenizer.eos_token, tokenizer.eos_token_id
            )
        )
        logger.info(
            "pad token: {}\tpad token id: {}".format(
                tokenizer.pad_token, tokenizer.pad_token_id
            )
        )
        logger.info("padding side: {}".format(tokenizer.padding_side))

        logger.info(
            "train_nums = {}, valid_nums = {}, test_nums = {}".format(
                train_nums,
                valid_nums,
                test_nums,
            )
        )
        logger.info(
            "num_gpus = {}, total_steps = {}, warmup_steps = {}, eval_steps = {}".format(
                num_gpus,
                total_steps,
                warmup_steps,
                eval_interval_steps,
            )
        )
        print("***** Running Training *****")

    # set `False` to be compatible with checkpointing
    model.config.use_cache = False

    if args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if train_mode == "sft":
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elif train_mode == "seq2seq":
        raise NotImplementedError("Seq2Seq training is not supported yet.")
    elif train_mode == "hpo":
        # perform hyperparameter search on the single rank
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="wandb",
            hp_space=create_wandb_hp_space,
            n_trials=30,
            compute_objective=lambda metrics: metrics["eval_sentence_bleu_cpp"],
        )
    else:
        raise ValueError(f"Invalid training mode: {train_mode}")

    # Prediction
    if do_predict and test_data is not None:
        if trainer.state.is_world_process_zero:
            logger.info("***** Running Prediction *****")
        test_metrics = trainer.predict(
            test_dataset=test_data,
            metric_key_prefix="test",
        ).metrics
        if trainer.state.is_world_process_zero:
            logger.info("test_metrics: {}".format(test_metrics))
            wandb.log(test_metrics)

    wandb.finish()

    logger.info("***** Checkpointing *****")
    model = unfreeze_all_layers(model)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)
    # save tokenizer for each detected checkpoint directory in output_dir
    for checkpoint_dir in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        try:
            tokenizer.save_pretrained(checkpoint_dir)
        except:
            pass

    print(
        "\n If there's a warning about missing keys above when using lora to train, please disregard :)"
    )
    logger.info("Training Success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_file", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, help="deepspeed config")
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="either training checkpoint or final adapter",
    )
    # parser.add_argument("--group_by_length", action="store_true", help="Faster, but produces an odd training loss curve,")
    parser.add_argument(
        "--lora_hyperparams_file",
        default="",
        type=str,
        help="Provide it when use_lora=True",
    )
    parser.add_argument(
        "--use_lora", action="store_true", default=False, help="Use lora"
    )
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    train(args)
