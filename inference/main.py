# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2023/11/09

import os
import math
import argparse
from typing import Optional, Tuple

from numpy import int64, float64
from rich.table import Table
from rich.console import Console
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from args import parse_arguments
from eval import compute_metric
from utils import (
    load_prompt,
    load_tokenizer,
    dump_json,
    dump_tsv,
)
from data_utils import (
    get_lexical_data,
    load_icl_example,
    postprocess,
    construct_prompt,
)
from model import (
    init_local_model,
    query_local_model,
    query_openai,
)

console = Console()


def round_float(num: float, n: int = 2) -> float:
    """
    Round float number with n decimal places

    Args:
    - num: float
    - n: int

    Returns:
    - rounded_num: float
    """
    return math.ceil(num * 10**n) / 10**n


def query_llm(
    args: argparse.Namespace,
    prompt: str,
    local_model: Optional[AutoModelForCausalLM],
    tok: AutoTokenizer,
) -> str:
    """
    Query LLM model with arguments and user prompt

    Args:
    - args: argparse.Namespace
    - prompt: str

    Returns:
    - response: str
    """
    model = args.model
    if "gpt-3.5-turbo" in model or "gpt-4" in model:
        try:
            tok = load_tokenizer(args.model)
        except Exception as _:
            tok = load_tokenizer("gpt-4-turbo")
        return query_openai(args, prompt, tok, console)
    elif "claude" in model:
        # return query_claude(args, prompt, console)
        tok = load_tokenizer("gpt-4-turbo")
        return query_openai(args, prompt, tok, console)
    elif "gemini" in model:
        # return query_gemini(args, prompt, console)
        tok = load_tokenizer("gpt-4-turbo")
        return query_openai(args, prompt, tok, console)
    elif args.run_local_model:
        tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        return query_local_model(args, prompt, console, local_model, tok)
    else:
        # use openai pathways as default
        tok = load_tokenizer("gpt-4-turbo")
        return query_openai(args, prompt, tok, console)


def request_llm(
    args: argparse.Namespace,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> None:
    """
    Request LLM model with arguments and user prompt

    Args:
    - args: argparse.Namespace
    - model: Optional[AutoModelForCausalLM]
    - tokenizer: Optional[AutoTokenizer]

    Returns:
    - None
    """
    task = args.task

    # load prompt template
    console.print(f"[bold green]Loading prompt template for {[task]} ...")
    prompt_template = load_prompt(prompt_path=args.prompt_path)
    # console.print(prompt_template)

    # load data for inference
    preds, golds = [], []
    console.print("[bold green]Loading data ...")
    if task in ["word-interpretation", "synthesize-context", "synthesize-definition"]:
        instances, words, word2refs = get_lexical_data(
            data_path=args.input_path,
            task_type=task,
            max_num_limit=args.max_query,
        )
    else:
        raise ValueError("Task type is not supported!")
    # console.print_json(data=instances)
    # exit()

    # load examples for few-shot demonstration
    examples = None
    if args.shot_num > 0:
        console.print("[bold green]Loading examples for few-shot demonstration ...")
        examples = load_icl_example(
            data_path=args.example_path,
            task_type=args.task,
            shot_num=args.shot_num,
            max_num_limit=args.max_query,
            retrieval_policy=args.retrieval_policy,
        )

    dataset = args.input_path.split("/")[1]
    os.makedirs(f"results/{dataset}", exist_ok=True)
    json_list_output = list()
    console.rule(title="[bold yellow]Start Running[/bold yellow]")
    with console.status(
        f"[bold yellow]{[args.model]} Running {task} ({args.shot_num}-shot) ..."
    ) as _:
        idx = 1

        if args.evaluate:
            if task == "word-interpretation":
                metric_sum_dict = {
                    # "bleu-hf": 0.0,
                    # "bleu-sacre": 0.0,
                    # "bleu-nltk": 0.0,
                    "bleu-cpp": 0.0,  # 1.
                    "rouge": 0.0,  # 2.
                    # "meteor": 0.0,  # 3.
                    # "exact-match": 0.0,
                    # "bert-score": 0.0,  # 4.
                    # "bleurt": 0.0,
                    # "mover-score": 0.0,  # 5.
                    # "mauve": 0.0,  # 6.
                }
            else:
                metric_sum_dict = {}

        is_ok = True
        batch_size = 128
        for i, instance in enumerate(instances):
            if args.retrieval_policy == "topk":
                prompt = construct_prompt(
                    args,
                    instance,
                    prompt_template,
                    examples[i],
                )
            else:
                prompt = construct_prompt(
                    args,
                    instance,
                    prompt_template,
                    examples,
                )
            console.print(prompt)
            # exit()
            # prompt = '"frozen with horror" What is the definition of "frozen"?'
            # TODO: consider to rewrite heurustic rules in the postprocess function
            if args.run_local_model:
                response = query_llm(
                    args,
                    prompt,
                    model,
                    tokenizer,
                )  # [(response, log_likelihood), ...]
                response, log_likelihood = zip(*response)
                response = list(response)
                log_likelihood = list(log_likelihood)
                response = postprocess(
                    response,
                    task,
                    args,
                )
            else:
                response = postprocess(
                    query_llm(args, prompt, model, tokenizer), task, args
                )
            # response = response.replace("[/INST]", "").replace("[INST]", "")
            if response == "":
                print("Detected empty response. Skip this instance.")
                idx += 1
                continue
            # console.log(response)
            # response = log_likelihood_estimator.compute(response)
            # console.log(response)
            # exit()
            # Display predictive result
            if task == "word-interpretation":
                if args.evaluate:
                    scores_dict = {
                        "bleu-cpp": [],
                        "rouge": [],
                    }
                    metric_dict = {
                        # "bleu-sacre": 0.0,
                        # "bleu-nltk": 0.0,
                        "bleu-cpp": 0.0,  # 1.
                        "rouge": 0.0,  # 2.
                        # "meteor": 0.0,  # 3.
                        # "exact-match": 0.0,
                        # "bert-score": 0.0,  # 4.
                        # "bleurt": 0.0,
                        # "mover-score": 0.0,  # 5.
                        # "mauve": 0.0,  # 6.
                    }
                    for k in metric_sum_dict.keys():
                        # try:
                        returns = compute_metric(
                            metric_name=k,
                            pred=response,
                            gold=(
                                instance["references"]
                                if "references" in instance
                                else instance["definition"]
                            ),
                            word=instance["word"],
                            refs=word2refs[instance["word"]],
                            dataset=dataset,
                        )
                        # except Exception as e:
                        # console.log(f"Error: {e}")
                        # is_ok = False
                        # break
                        # try:
                        # console.log(returns)
                        if isinstance(returns, int64):
                            score = returns[0]
                            scores = []
                        elif isinstance(returns, float64):
                            score = returns
                            scores = []
                        elif isinstance(returns, Tuple):
                            score = float(returns[0])
                            scores = returns
                        else:
                            score = float(returns)
                            scores = [returns]
                        # elif len(returns) == 1:
                        # score = returns[0]
                        # scores = []
                        # elif len(returns) == 2:
                        # score, scores = returns[0], returns[1]
                        # if score < 0:
                        # is_ok = False
                        # break
                        # assert len(scores) == len(response) == len(log_likelihood)
                        # except Exception as e:
                        # console.log(f"Error: {e}")
                        # is_ok = False
                        # break
                        # else:
                        # console.log(f"{k}: {score}")
                        # console.log(f"{k}: {scores}")
                        # pass
                        metric_dict[k] = (
                            score
                            if isinstance(score, float) or isinstance(score, int)
                            else max(score)
                        )
                        metric_sum_dict[k] += metric_dict[k]
                        scores_dict[k] = scores
                    if not is_ok:
                        console.log("Skip this instance.")
                        continue
                    if args.verbal:
                        try:
                            console.print(
                                f"[bold gold1]\\[Term: Context] {instance['word']}: [underline]{instance['context']}[/underline][/bold gold1]\n"
                                f"\t[bold purple]\\[Gold] {instance['definition']} [/bold purple]\n"
                                f"[bold dark_olive_green3]\\[Pred] {response}[/bold dark_olive_green3]\n"
                                # f"\t[bold orange_red1]\\[BLEU-NLTK] {round((metric_sum_dict['bleu-nltk'] / idx) * 100, 2)} [/bold orange_red1]"
                                f"\t[bold orange_red1]\\[BLEU] {round_float(metric_sum_dict['bleu-cpp'] / idx)} [/bold orange_red1]"
                                # f"\t[bold orange_red1]\\[BLEU-Sacre] {round(metric_sum_dict['bleu-sacre'] / idx, 2)} [/bold orange_red1]"
                                f"\t[bold orange_red1]\\[RougeL] {round_float(metric_sum_dict['rouge'] / idx)} [/bold orange_red1]"
                                # f"\t[bold orange_red1]\\[Meteor] {round_float(metric_sum_dict['meteor'] / idx * 100)}[/bold orange_red1]"
                                # f"\t[bold orange_red1]\\[ExactMatch] {round((metric_sum_dict['exact-match'] / idx) * 100, 2)}[/bold orange_red1]"
                                # f"\t[bold orange_red1]\\[BertScore] {round_float(metric_sum_dict['bert-score'] / idx * 100)}[/bold orange_red1]"
                                # f"\t[bold orange_red1]\\[BLEURT] {round(metric_sum_dict['bleurt'] / idx, 2) * 100}[/bold orange_red1]"
                                # f"\t[bold orange_red1]\\[MoverScore] {round_float(metric_sum_dict['mover-score'] / idx)}[/bold orange_red1]"
                                # f"\t[bold orange_red1]\\[Mauve] {round_float(metric_sum_dict['mauve'] / idx * 100)}[/bold orange_red1]"
                            )
                        except Exception as _:
                            pass
                else:
                    if args.verbal:
                        console.print(
                            f"[bold dark_olive_green3]\\[Pred] {instance['word']}: [underline]{response}[/underline][/bold dark_olive_green3]"
                            f"\t[bold purple]\\[Gold] {instance['definition']} [/bold purple]"
                        )
            if task == "synthesize-context":
                response_display = (
                    response.replace(
                        instance["word"], f"[underline]{instance['word']}[/underline]"
                    )
                    .replace(
                        instance["word"].capitalize(),
                        f"[underline]{instance['word'].capitalize()}[/underline]",
                    )
                    .replace("###\n", "\n- ")
                    .replace("###", "\n- ")
                )
                if response_display.endswith("\n- "):
                    response_display = response_display[:-3]
                context_display = instance["context"].replace(
                    instance["word"], f"[underline]{instance['word']}[/underline]"
                )
            if task == "synthesize-definition":
                response_display = (
                    response.replace(
                        instance["word"], f"[underline]{instance['word']}[/underline]"
                    )
                    .replace(
                        instance["word"].capitalize(),
                        f"[underline]{instance['word'].capitalize()}[/underline]",
                    )
                    .replace("###\n", "\n- ")
                    .replace("###", "\n- ")
                )
                if response_display.endswith("\n- "):
                    response_display = response_display[:-3]
                definition_display = instance["definition"].replace(
                    instance["word"], f"[underline]{instance['word']}[/underline]"
                )
                if args.verbal:
                    try:
                        pass
                        # console.print(
                        # f"[bold dark_olive_green3]\[Synthesis]\n- {response_display}[/bold dark_olive_green3]\n[bold purple]\[Reference]\n- {definition_display}[/bold purple]"
                        # )
                    except:
                        pass
                        # console.print(
                        # f"[bold dark_olive_green3]\[Pred] {instance['word']}: [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Ref] [underline] {instance['definition']}[/underline][/bold purple]"
                        # )

            console.log(f"Query {idx} completed.")
            idx += 1
            if task == "synthesize-context":
                instance["synthetic_context"] = [
                    s.strip()
                    for s in response.split("###")
                    if s.strip() != "" and "Here are" not in s
                ]
            elif task == "synthesize-definition":
                instance["synthetic_definition"] = [
                    s.strip()
                    for s in response.split("###")
                    if s.strip() != "" and "Here are" not in s
                ]
            else:
                instance["prediction"] = response
                # instance["log_probs"] = log_likelihood
                # instance["bleu"] = scores_dict["bleu-cpp"]
                # instance["rouge"] = scores_dict["rouge"]

            json_list_output.append(instance)

            # dump data
            if (batch_size > 0 and idx % batch_size == 0) or idx == len(instances) - 1:
                if "json" in args.output_format:
                    if "train" in args.input_path:
                        output_path = f"results/{dataset}/{args.task}-{args.model.replace('/', '-')}.train.json".replace(
                            "--", "-"
                        )
                    elif "test" in args.input_path:
                        output_path = f"results/{dataset}/{args.task}-{args.model.replace('/', '-')}.test.json".replace(
                            "--", "-"
                        )
                    else:
                        output_path = f"results/{dataset}/{args.task}-{args.model.replace('/', '-')}.json".replace(
                            "--", "-"
                        )
                    dump_json(output_path, json_list_output, indent=True)

        console.rule(title="[bold yellow]End Running[/bold yellow]")

    console.print()
    console.rule(title="[bold yellow]Running Report[/bold yellow]")
    # TODO: add performance reporter
    table = Table(
        title="Running Reporter", show_header=True, header_style="bold magenta"
    )
    table.add_column("Model")
    table.add_column("Task")
    # table.add_column("Shot Num")
    table.add_column("N")
    for k in metric_sum_dict.keys():
        table.add_column(k)
    table.add_row(
        args.model,
        task,
        # str(args.shot_num),
        str(args.num_return_sequences),
        *[
            str(
                # round(metric_sum_dict[k] / idx, 3) * 100
                math.ceil(metric_sum_dict[k] / idx * 100) / 100
            )
            for k in metric_sum_dict.keys()
        ],
    )
    console.print(table)
    console.rule(title="")


if __name__ == "__main__":
    # parsing input arguments
    args = parse_arguments()
    if args.model.endswith("/"):
        args.model = args.model.rsplit("/", 1)[0].strip()
    if args.model in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
        tok = load_tokenizer(args.model)
    else:
        # load GPT-2 Tokenizer as default
        tok = AutoTokenizer.from_pretrained("gpt2")

    model, tokenizer = None, None
    if args.run_local_model:
        model_type = (
            "sparse" if ("xLlama" in args.model or "ffn" in args.model) else "dense"
        )
        model, tokenizer = init_local_model(args.model, console, model_type=model_type)

    request_llm(args, model, tokenizer)  # request GPT-3.5-turbo / GPT-4
