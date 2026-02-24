import json
from time import time
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import OrderedDict

import torch
import bitsandbytes as bnb
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from datasets import Dataset
from transformers.trainer_pt_utils import get_parameter_names
from transformers import (
    TrainerCallback,
    ProgressCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    PreTrainedModel,
    PreTrainedTokenizer,
    EarlyStoppingCallback,
)
from transformers.utils import logging

IGNORE_INDEX = -100

logger = logging.get_logger(__name__)


class EarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(
        self,
        early_stopping_patience: int = 10,
        early_stopping_threshold: Optional[float] = 0.001,
    ):
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )


class TrackGPUUtilizationCallback(ProgressCallback):
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            self.training_bar = tqdm(
                total=state.max_steps,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
        self.current_step = 0

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            seqence_length = 128
            samples_num = args.per_device_train_batch_size * args.world_size
            tokens_num = seqence_length * samples_num
            if "last_timestamp" not in state.__dict__:
                state.last_timestamp = time()
            else:
                current_timestamp = time()
                time_diff = current_timestamp - state.last_timestamp
                state.last_timestamp = current_timestamp

                samples_per_sec = samples_num // time_diff
                tokens_per_sec = tokens_num // time_diff

                # nvmlInit()
                # handle = nvmlDeviceGetHandleByIndex(0)
                # info = nvmlDeviceGetMemoryInfo(handle)
                # gpu_memory = info.free // 1024 // 1024
                self.training_bar.update(1)
                self.training_bar.set_postfix(
                    {
                        "samples/sec": f"{samples_per_sec:,}",
                        "tokens/sec": f"{tokens_per_sec:,}",
                        # "gpu_memory": f"{gpu_memory:,} MB",
                    }
                )
                self.current_step = state.global_step


class TrackEvalResultCallback(TrainerCallback):
    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        super().on_evaluate(args, state, control, **kwargs)
        if "past_eval_results" not in state.__dict__:
            state.past_eval_results = []
        if "eval_count" not in state.__dict__:
            state.eval_count = 0

        state.past_eval_results.append(metrics)
        # print(f"Eval results: {metrics}")

        # Save the evaluation results if current process is the global main process
        if state.is_world_process_zero:
            save_dir, pred_str, label_str = state.eval_pred
            # time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # remove directory first if exists
            with open(f"{save_dir}/eval-{str(state.eval_count)}.tsv", "w") as f:
                f.write("Prediction\tReference\n")
                for pred, label in zip(pred_str, label_str):
                    f.write(f"{' '.join(pred.split())}\t{' '.join(label.split())}\n")
                f.write("\n")
                # f.write(f"seen tokens\t{state.num_input_tokens_seen}\n")
                f.write(f"total flos\t{state.total_flos}\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}\t{str(value)}\n")

            inputs, outputs, labels = state.inputs, state.outputs, state.labels
            with open(f"{save_dir}/io-{str(state.eval_count)}.txt", "w") as f:
                f.write("Input\tOutput\tLabel\n")
                for input_, output, label in zip(inputs, outputs, labels):
                    f.write(
                        f"{' '.join(input_.split())}\t{' '.join(output.split())}\t{' '.join(label.split())}\n"
                    )

        state.eval_count += 1


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        logger.info(f"***** Running Evaluation *****")
        if state.global_step == 0:
            control.should_evaluate = True


def get_adam_8bit_optimizer(model: PreTrainedModel, training_args: TrainingArguments):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_8bit_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )

    return adam_8bit_optim


def get_adam_optimizer(model: PreTrainedModel, training_args: TrainingArguments):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    optimizer = torch.optim.Adam(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )

    return optimizer


def get_lr_scheduler(
    optimizer: Any,
    training_steps: int,
    warmup_ratio: float,
    lr_begin: float = 1e-6,
    lr_end: float = 5e-7,
):
    """
    Creates a linear learning rate scheduler with warmup for PyTorch optimizer.

    Args:
        optimizer (Optimizer): PyTorch optimizer object.
        training_steps (int): Total number of training steps.
        warmup_ratio (float): Ratio of warmup steps.
        lr_begin (float, optional): Initial learning rate. Defaults to 1e-6.
        lr_end (float, optional): Final learning rate. Defaults to 5e-7.

    Returns:
        LambdaLR: Linear learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < warmup_ratio * training_steps:
            return (lr_end - lr_begin) / (
                warmup_ratio * training_steps
            ) * current_step + lr_begin
        else:
            return lr_end + (lr_begin - lr_end) * (
                current_step - warmup_ratio * training_steps
            ) / ((1 - warmup_ratio) * training_steps)

    return LambdaLR(optimizer, lr_lambda)


def dump_hf_dataset(data: Dataset, save_path: str, save_format: str = "jsonl") -> None:
    """
    Dumps a Hugging Face dataset to a JSONL file.

    Args:
        data (Dataset): Hugging Face dataset object.
        save_path (str): Path to save the JSONL file.
    """
    # drop ["input_ids", "attention_mask", "labels"]
    data = data.remove_columns(["input_ids", "attention_mask", "labels"])
    if save_format == "jsonl":
        with open(save_path, "w") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    else:
        raise NotImplementedError(f"Unsupported save format: {save_format}")


def aggregate_definitions(dataset: Dataset) -> OrderedDict[str, List[str]]:
    """
    Aggregates multiple definitions into a single word entry.

    Args:
        dataset (Dataset): Hugging Face dataset object.

    Returns:
        Dict[str, List[str]]: Dictionary of word definitions.
    """
    word2defs = OrderedDict()
    for example in dataset:
        word = example["term"]
        definition = example["definition"]
        if word not in word2defs:
            word2defs[word] = []
        word2defs[word].append(definition)

    return word2defs
