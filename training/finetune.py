# -*- coding: utf-8 -*-
#
# @author: Yang Liu <yangliu.real@gmail.com>
# @date: 2024/04/03
#
# LM-Lexicon: Towards Robust Lexical Definition Modeling

import os
import sys
import json
import glob
import argparse
import warnings
from ast import literal_eval
from functools import partial
from typing import Any, List, Dict, Optional, Tuple

import torch
import wandb
import paddlenlp
import datasets
import numpy as np
import transformers

from retry import retry
from rich.console import Console
from datasets import load_dataset
from nltk.tokenize import word_tokenize

from liger_kernel.transformers import (
    apply_liger_kernel_to_llama,
    apply_liger_kernel_to_gemma2,
    apply_liger_kernel_to_qwen3,
)
from moe.models.modeling_llama import LlamaForCausalLM as MoELlamaForCausalLM

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from eval import (
    compute_hf_bleu,
    compute_sacre_bleu,
    compute_google_bleu,
    compute_sentence_bleu,
    compute_nist,
    compute_rouge,
    compute_meteor,
    compute_moverscore,
    compute_bert_score,
    compute_mauve,
    compute_exact_match,
    compute_edit_distance,
    compute_distinct,
    compute_sari,
    compute_generation_length,
    get_word2refs,
)
from utils import (
    IGNORE_INDEX,
    EarlyStoppingCallback,
    TrackGPUUtilizationCallback,
    EvaluateFirstStepCallback,
    TrackEvalResultCallback,
    dump_hf_dataset,
    get_lr_scheduler,
    get_adam_optimizer,
    get_adam_8bit_optimizer,
    postprocess_text,
    unfreeze_all_layers,
    aggregate_definitions,
    optimize_specific_layers,
    create_sft_trainer,
    get_logger,
    get_dataset_path_prefix,
)

# set up wandb
os.environ["WANDB_SILENT"] = "false"
os.environ["WANDB_DISABLED"] = "false"
os.environ["WANDB_ENTITY"] = "definition-modeling"
os.environ["WANDB_PROJECT"] = "dm"
os.environ["WANDB_CACHE_DIR"] = os.path.expanduser("~/.cache/wandb")
os.environ["WANDB_API_KEY"] = "6dee9586d0042e7fc6a76b0be341b9ba2650d3b1"


# set up logging
console = Console()
warnings.filterwarnings("ignore")
# transformers.logging.set_verbosity_info()


def train(args: argparse.Namespace) -> None:
    """Training entry for supervised fine-tuning.

    Args:
    - args(argparse.Namespace): input arguments

    Returns:
    - None
    """
    model_config = json.load(open(args.model_config, "r"))
    do_train = True if literal_eval(model_config["do_train"]) else False
    do_eval = True if literal_eval(model_config["do_eval"]) else False
    do_predict = True if literal_eval(model_config["do_predict"]) else False
    model_type = model_config["model_type"]
    model_name_or_path = model_config["model_name_or_path"]
    train_mode = "sft"
    mask_term = True if model_config["mask_term"] == "True" else False
    data_path_train = model_config["data_path_train"]
    data_path_valid = model_config["data_path_valid"]
    data_path_test = model_config["data_path_test"]
    learning_rate = model_config["learning_rate"]
    eval_samples = model_config["eval_samples"]
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

    # """
    wandb.init(
        project="dm",
        name=run_name,
        resume="must" if args.resume_from_checkpoint else None,
        id="3u3t7ril",
    )
    # """

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
    if model_type.lower() == "qwen3":
        # replace_llama_forward_with_emo_1_adaptive_forward(logger, model_type="dense")
        # replace_llama_forward_with_emo_2_fixed_forward(logger, model_type="dense")
        # automatically monkey-patches the model with the optimized Liger kernels
        apply_liger_kernel_to_qwen3()
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
        )
        tokenizer.bos_token_id = 151643
        tokenizer.eos_token_id = 151645
        tokenizer.pad_token_id = 151643
        # tokenizer.pad_token_id = 151662
    elif model_type.lower() == "llama-3":
        # replace_llama_forward_with_emo_1_adaptive_forward(logger, model_type="dense")
        # replace_llama_forward_with_emo_2_fixed_forward(logger, model_type="dense")
        # automatically monkey-patches the model with the optimized Liger kernels
        apply_liger_kernel_to_llama()
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
        )
        tokenizer.bos_token_id = 128000
        tokenizer.eos_token_id = 128001
        # tokenizer.pad_token_id = tokenizer.bos_token_id
        tokenizer.pad_token_id = 128009
    elif model_type.lower() == "llama-moe":
        # replace_llama_forward_with_emo_1_adaptive_forward(logger, model_type="sparse")
        logger.info("Using MoE-LLaMA model for training.")
        # automatically monkey-patches the model with the optimized Liger kernels
        apply_liger_kernel_to_llama()
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
        )
        tokenizer.bos_token_id = 128000
        tokenizer.eos_token_id = 128001
        # tokenizer.pad_token_id = tokenizer.bos_token_id
        tokenizer.pad_token_id = 128009
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

    # @timeit
    @retry(tries=3, delay=10, backoff=2)
    def compute_metrics(eval_pred: Tuple[Any], compute_result: bool = True):
        """Compute metrics for evaluation."""

        metrics = {
            # "bleu": 0.0,
            # "sacrebleu": 0.0,
            # "google_bleu": 0.0,
            # "sentence_bleu": 0.0,
            "sentence_bleu_cpp": 0.0,
            "rouge-l": 0.0,
            # "mover-score": 0.0,
            # "meteor": 0.0,
            # "bertscore-p": 0.0,
            # "bertscore-r": 0.0,
            # "bertscore-f1": 0.0,
            # "mauve": 0.0,
            # "exact_match": 0.0,
            # "edit_distance": 0.0,
            # "sari": 0.0,
            # "nist": 0.0,
            # "distinct-1": 0.0,
            # "distinct-2": 0.0,
            # "distinct-3": 0.0,
            "generation_length": 0.0,
        }
        if trainer.state.is_world_process_zero:
            if trainer.args.include_inputs_for_metrics:
                preds, labels, inputs = eval_pred
                inputs = np.where(
                    inputs != IGNORE_INDEX, inputs, tokenizer.pad_token_id
                )
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
            preds = np.where(
                preds != IGNORE_INDEX, preds, tokenizer.pad_token_id
            ).astype(int)
            labels = np.where(
                labels != IGNORE_INDEX, labels, tokenizer.pad_token_id
            ).astype(int)

            # heuristic rules-based post-processing
            to_process_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            to_process_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            inputs_str = tokenizer.batch_decode(inputs, skip_special_tokens=True)
            preds_str, labels_str = postprocess_text(
                to_process_preds, to_process_labels
            )

            # extract word from each input samples
            words = [
                # s.replace('"?', "").split('"')[-1].split("\n\n", maxsplit=1)[0].strip()
                s.split('"?\n\n', maxsplit=1)[0].rsplit('"', maxsplit=1)[-1].strip()
                for s in inputs_str
            ]
            # if len(words) != len(preds_str) != len(labels_str):
            # global valid_words
            # words = valid_words[: len(preds_str)]
            # words = [word for word in words if word in valid_words]

            # update state with save_dir, words, inputs, outputs, and labels
            trainer.state.save_dir = f"{output_dir}/eval"
            trainer.state.words = words
            trainer.state.inputs = inputs_str
            trainer.state.outputs = preds_str
            trainer.state.labels = labels_str

            assert len(preds_str) == len(labels_str) == len(inputs_str) == len(words), (
                f"Length mismatch between `preds_str`, `labels_str`, `inputs_str`, and `words` "
                f"in evaluation: {len(preds_str)}, {len(labels_str)}, {len(inputs_str)}, {len(words)}"
            )

            # compute BLEU
            # metrics["bleu"] = compute_hf_bleu(
            # preds=preds_str,
            # refs=labels_str,
            # )

            # # compute SacreBLEU
            # metrics["sacrebleu"] = compute_sacre_bleu(
            # preds=preds_str,
            # refs=labels_str,
            # )

            # # compute Google BLEU
            # metrics["google_bleu"] = compute_google_bleu(
            # preds=preds_str,
            # refs=labels_str,
            # )

            # compute sentence BLEU (nltk)
            # metrics["sentence_bleu"] = compute_sentence_bleu(
            # preds=preds_str,
            # refs=labels_str,
            # mode="nltk",
            # words=words,
            # word2refs=word2refs,
            # )

            # compute sentence BLEU (cpp)
            metrics["sentence_bleu_cpp"] = compute_sentence_bleu(
                preds=preds_str,
                refs=labels_str,
                mode="cpp",
                words=words,
                word2refs=word2refs,
            )

            # compute sentence ROUGE with multiple references
            metrics["rouge-l"] = compute_rouge(
                preds=preds_str,
                refs=labels_str,
                words=words,
                word2refs=word2refs,
                metric_key="rougeL",
            )

            # compute sentence MOVER with multiple references
            # metrics["mover-score"] = compute_moverscore(
            # preds=preds_str,
            # refs=labels_str,
            # words=words,
            # word2refs=word2refs,
            # )

            # # compute METEOR
            # metrics["meteor"] = compute_meteor(
            # preds=preds_str,
            # refs=labels_str,
            # words=words,
            # word2refs=word2refs,
            # )

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
            # metrics["bertscore-p"], metrics["bertscore-r"], metrics["bertscore-f1"] = (
            # compute_bert_score(
            # preds=preds_str,
            # refs=labels_str,
            # words=words,
            # word2refs=word2refs,
            # metric_key="all",
            # )
            # )

            # compute MAUVE
            # metrics["mauve"] = compute_mauve(
            # preds=preds_str,
            # refs=labels_str,
            # words=words,
            # word2refs=word2refs,
            # )

            # compute EXACT MATCH
            # metrics["exact_match"] = compute_exact_match(
            # preds=preds_str,
            # refs=labels_str,
            # words=words,
            # word2refs=word2refs,
            # )

            # compute EDIT DISTANCE
            # metrics["edit_distance"] = compute_edit_distance(
            # preds=preds_str,
            # refs=labels_str,
            # ignore_case=True,
            # )

            # compute SARI
            # metrics["sari"] = compute_sari(
            # sources=inputs_str,
            # preds=preds_str,
            # refs=labels_str,
            # words=words,
            # word2refs=word2refs,
            # )

            # compute NIST
            # metrics["nist"] = compute_nist(
            # preds_str,
            # labels_str,
            # words=words,
            # word2refs=word2refs,
            # )

            # compute DISTINCT-1, DISTINCT-2, and DISTINCT-3
            # metrics["distinct-1"], metrics["distinct-2"], metrics["distinct-3"] = (
            # compute_distinct(preds_str)
            # )

            # compute generation length
            metrics["generation_length"] = compute_generation_length(preds_str)

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

        full_text = input_text + target_text

        if tokenize_train:
            tokenized_full_sequence = tokenize(full_text)
            if not train_on_input:
                tokenized_input_prompt = tokenize(input_text)
                input_prompt_len = len(tokenized_input_prompt["input_ids"])
                tokenized_full_sequence["labels"] = [
                    -100
                ] * input_prompt_len + tokenized_full_sequence["labels"][
                    input_prompt_len:
                ]
                tokenized_full_sequence["attention_mask"] = [1] * len(
                    tokenized_full_sequence["input_ids"]
                )
        else:
            tokenized_input_sequence = tokenize(input_text)  # input ids
            tokenized_target_sequence = tokenize(target_text)  # target ids
            tokenized_full_sequence = tokenize(full_text)  # full ids

            # method 1. misalign the input and target sequences (we can not compute the loss)
            # issue: the sequence sizes of input and target are misaligned, throw error ❌
            """
            tokenized_full_sequence["input_ids"] = tokenized_input_sequence["input_ids"]
            tokenized_full_sequence["labels"] = tokenized_target_sequence["input_ids"]
            tokenized_full_sequence["attention_mask"] = [1] * len(
                tokenized_input_sequence["input_ids"]
            )
            """
            # method 2. left padding for input and right padding for target sequence
            # issue: the sequence sizes of input and target are aligned, we can not compute eval loss ⚠️
            tokenized_full_sequence["input_ids"] = [tokenizer.pad_token_id] * (
                max_seq_len - len(tokenized_input_sequence["input_ids"])
            ) + tokenized_input_sequence["input_ids"]
            tokenized_full_sequence["labels"] = tokenized_target_sequence[
                "input_ids"
            ] + [tokenizer.pad_token_id] * (
                max_seq_len - len(tokenized_target_sequence["input_ids"])
            )
            tokenized_full_sequence["attention_mask"] = [0] * (
                max_seq_len - len(tokenized_input_sequence["input_ids"])
            ) + [1] * len(tokenized_input_sequence["input_ids"])

        return tokenized_full_sequence

    def generate_and_tokenize_prompt(
        datapoint: Dict[str, Any], tokenize_train: bool
    ) -> Dict[str, str]:
        """Generate and construct prompt constrained by a fixed size of window.
        Dynamically generate input sequence and target sequence for each training example.

        Args:
        - datapoint(Dict[str, Any]): datapoint

        Returns:
        - tokenized result
        """
        term = datapoint["term"].strip()
        input_text = tokenizer.bos_token + datapoint["instruction"].strip() + "\n\n"
        target_text = datapoint["definition"].strip() + tokenizer.eos_token
        input_size = len(
            tokenizer.encode(input_text + target_text, add_special_tokens=True)
        )
        if input_size <= max_seq_len:
            # tokenize and return the input_text and target_text
            tokenized_text = tokenize_full_sequence(
                input_text, target_text, term, tokenize_train=tokenize_train
            )
            return tokenized_text
        else:
            # drop the data point if the length of input_text and target_text is greater than max_seq_len
            # console.log(
            # f"[bold yellow]Drop the data point due to the length of input_text and target_text is greater than {max_seq_len}.[/bold yellow]"
            # )
            return {"input_ids": [], "attention_mask": [], "labels": []}

    if use_lora:
        model = prepare_model_for_kbit_training(model)
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
            data_train.shuffle(keep_in_memory=False, seed=42)
            .map(
                lambda x: {**x, "term": x["term"].strip()},
                num_proc=1,
                keep_in_memory=False,
                load_from_cache_file=True,
            )
            .map(
                partial(generate_and_tokenize_prompt, tokenize_train=True),
                num_proc=1,
                keep_in_memory=False,
                load_from_cache_file=True,
            )
            .filter(
                lambda x: len(x["input_ids"]) > 0
                and len(x["input_ids"]) <= max_seq_len,
                num_proc=1,
                keep_in_memory=False,
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
        unique_terms = set()
        val_data = (
            data_valid.map(
                lambda x: {**x, "term": x["term"].strip()},
                num_proc=1,
                keep_in_memory=False,
                load_from_cache_file=True,
            )
            .map(
                partial(generate_and_tokenize_prompt, tokenize_train=False),
                num_proc=1,
                keep_in_memory=False,
                load_from_cache_file=True,
            )
            .filter(
                lambda x: len(x["input_ids"]) > 0
                and len(x["input_ids"]) <= max_seq_len,
                num_proc=1,
                keep_in_memory=False,
            )
            # .filter(
            # lambda x: x["term"] not in unique_terms
            # and not unique_terms.add(x["term"]),
            # num_proc=1,
            # keep_in_memory=False,
            # )
        )
        global valid_words
        valid_words = [example["term"].strip() for example in val_data]
        if isinstance(eval_samples, int):
            val_data = (
                val_data.select(range(eval_samples), keep_in_memory=False)
                if eval_samples <= len(val_data)
                else val_data
            )
            valid_words = [example["term"].strip() for example in val_data]
            # dump validation data
            dump_hf_dataset(val_data, save_path=f"{output_dir}/eval/valid.jsonl")
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
            # data_test.shuffle(keep_in_memory=False, seed=42)
            data_test.map(
                lambda x: {**x, "term": x["term"].strip()},
                num_proc=1,
                keep_in_memory=False,
                load_from_cache_file=True,
            )
            .map(
                partial(generate_and_tokenize_prompt, tokenize_train=False),
                num_proc=1,
                keep_in_memory=False,
                load_from_cache_file=True,
            )
            .filter(
                lambda x: len(x["input_ids"]) > 0
                and len(x["input_ids"]) <= max_seq_len,
                num_proc=1,
                keep_in_memory=False,
            )
        )
        # global valid_words
        # valid_words = [example["term"].strip() for example in test_data]
        if isinstance(eval_samples, int):
            # don't select test data
            test_data = (
                test_data.select(range(eval_samples), keep_in_memory=False)
                if eval_samples <= len(test_data)
                else test_data
            )
            # valid_words = [example["term"].strip() for example in test_data]
            # dump test data
            dump_hf_dataset(test_data, save_path=f"{output_dir}/eval/test.jsonl")
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

    # concate validation and test data for evaluation
    dataset_path_prefix = get_dataset_path_prefix(data_path_train)
    # dataset_to_refer = data_test
    dataset_to_refer = datasets.concatenate_datasets([data_valid, data_test])
    # construct word to multiple references mapping
    word2refs = get_word2refs(
        words=dataset_to_refer["term"],
        refs=dataset_to_refer["definition"],
        # dedup_refs=False,
        dedup_refs=True,
        cache_path=f"{dataset_path_prefix}/word2refs.json",
    )

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
        # early_stopping=False,
        do_sample=True,
        repetition_penalty=1.05,
        temperature=0.6,
        top_p=0.9,
        num_return_sequences=1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
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
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"num_cycles": 0.5, "min_lr_rate": 0.1},
        seed=42,
        data_seed=42,
        # fp16=True,  # use fp16 (half precision) for training as default
        bf16=True,  # ad hoc for Llama-2,3
        tf32=True,  # ad hoc for NVIDIA Ampere GPUs
        bf16_full_eval=True,  # use bf16 for full evaluation for Llama 3
        # enable gradient(activation) checkpointing as default to save GRAM
        eval_on_start=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        log_level="info",
        logging_first_step=True,
        logging_dir="logs/tensorboard",
        logging_steps=model_config["logging_steps"],
        eval_strategy="steps",
        eval_steps=eval_interval_steps,
        save_steps=save_interval_steps,
        # disable_tqdm=True,
        disable_tqdm=False,
        save_safetensors=True,
        # eval_accumulation_steps=16,
        # evaluation_strategy="epoch",
        # save_strategy="epoch",
        output_dir=output_dir,
        overwrite_output_dir=True,  # ⚠️  Whether to overwrite the output directory
        save_total_limit=3,
        save_only_model=True,
        include_inputs_for_metrics=True,
        include_num_input_tokens_seen=True,
        batch_eval_metrics=False,
        predict_with_generate=True,  # ⚠️  Whether to use generate to calculate generative metrics (ROUGE, BLEU).
        metric_for_best_model="sentence_bleu_cpp",  # bleu, sacrebleu, google_bleu, sentence_bleu, sentence_bleu_cpp, etc.
        greater_is_better=True,
        # load_best_model_at_end=True,
        # dataloader_drop_last=True,
        dataloader_drop_last=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        use_liger_kernel=True,
        # used to avoid GPU socket timeouts when performing slow operations in distributed runnings
        ddp_timeout=3600,  # 60 mins
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
    # adam_8bit_optim = get_adam_8bit_optimizer(model, training_args)
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
            optimizer=None,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics,
            callbacks=[
                # TrackGPUUtilizationCallback,
                # EarlyStoppingCallback,
                # EvaluateFirstStepCallback,
                TrackEvalResultCallback,
            ],
            data_collator=data_collator,
            device_map=device_map,
        )
    elif train_mode == "seq2seq":
        raise NotImplementedError("Seq2Seq training is not supported yet.")
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
        # Placeholder for hyperparameter space - HPO mode requires implementation
        create_wandb_hp_space = lambda: {}
        # perform hyperparameter search on the single rank
        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="wandb",
            hp_space=create_wandb_hp_space,
            n_trials=30,
            compute_objective=lambda trial: trial["eval_sentence_bleu_cpp"],
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
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, help="deepspeed config")
    parser.add_argument(
        "--resume_from_checkpoint",
        # action="store_true",
        type=str,
        required=False,
        default=None,
        help="either training checkpoint or final adapter",
    )
    # parser.add_argument("--group_by_length", action="store_true", help="Faster, but produces an odd training loss curve,")
    parser.add_argument(
        "--use_lora", action="store_true", default=False, help="Use lora"
    )
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    train(args)
