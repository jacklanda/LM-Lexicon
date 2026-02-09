# -*- coding: utf-8 -*-
#
# @Author: Yang Liu <yangliu.real@gmail.com>
# @Date: 2023/11/09

import os
import sys
import time
import argparse
from typing import Optional

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
    get_wsd_data,
    load_icl_example,
    postprocess,
    construct_prompt,
)
from model import (
    init_local_model,
    query_local_model,
    query_openai,
    query_claude,
    query_gemini,
    AVAILABLE_MODELS,
)

console = Console()


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
        tok = load_tokenizer(args.model)
        return query_openai(args, prompt, tok, console)
    elif "claude" in model:
        return query_claude(args, prompt, console)
    elif model == "gemini-pro":
        return query_gemini(args, prompt, console)
    elif args.run_local_model:
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
        instances = get_lexical_data(
            data_path=args.input_path,
            task_type=task,
            max_num_limit=args.max_query,
        )
    elif task == "word-sense-disambiguation":
        instances = get_wsd_data(
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
            max_num_limit=args.shot_num,
        )

    os.makedirs("results", exist_ok=True)
    json_list_output = list()
    console.rule(title="[bold yellow]Start Running[/bold yellow]")
    with console.status(
        f"[bold yellow]{[args.model]} Running {task} ({args.shot_num}-shot) ..."
    ) as _:
        idx = 1

        if args.evaluate:
            if task == "word-interpretation":
                metric_sum_dict = {
                    "bleu": 0.0,
                    "rouge": 0.0,
                    "bert-score": 0.0,
                    # "bleurt": 0.0,
                    "meteor": 0.0,
                    "exact-match": 0.0,
                }
            elif task == "word-sense-disambiguation":
                metric_sum_dict = {
                    "exact-match": 0.0,
                }
            else:
                metric_sum_dict = {}

        for instance in instances:
            prompt = construct_prompt(
                args,
                instance,
                prompt_template,
                examples,
            )
            # console.print(prompt)
            # exit()
            # prompt = "\"Here is a heavy rain today.\" What is the definition of \"rain\"?"
            # TODO: consider to rewrite heurustic rules in the postprocess function
            response = postprocess(
                query_llm(args, prompt, model, tokenizer), task, args
            )
            # response = response.replace("[/INST]", "").replace("[INST]", "")
            if response == "":
                print("Detected empty response. Skip this instance.")
                continue
            # Display predictive result
            if task == "word-interpretation":
                if args.evaluate:
                    metric_dict = {
                        "bleu": 0.0,
                        "rouge": 0.0,
                        "bert-score": 0.0,
                        # "bleurt": 0.0,
                        "meteor": 0.0,
                        "exact-match": 0.0,
                    }
                    for k in metric_sum_dict.keys():
                        metric_dict[k] = compute_metric(
                            metric_name=k,
                            pred=response,
                            gold=(
                                instance["references"]
                                if "references" in instance
                                else instance["definition"]
                            ),
                        )
                        metric_sum_dict[k] += metric_dict[k]
                    if args.verbal:
                        console.print(
                            f"[bold dark_olive_green3]\\[Pred] {instance['word']}: [underline]{response}[/underline][/bold dark_olive_green3]"
                            f"\t[bold purple]\\[Gold] {instance['definition']} [/bold purple]"
                            f"\t[bold orange_red1]\\[BLEU] {round((metric_sum_dict['bleu'] / idx) * 100, 2)} [/bold orange_red1]"
                            f"\t[bold orange_red1]\\[Rouge-L] {round((metric_sum_dict['rouge'] / idx) * 100, 2)} [/bold orange_red1]"
                            f"\t[bold orange_red1]\\[BertScore] {round((metric_sum_dict['bert-score'] / idx) * 100, 2)}[/bold orange_red1]"
                            # f"\t[bold orange_red1]\\[BLEURT] {round(metric_sum_dict['bleurt'] / idx, 2) * 100}[/bold orange_red1]"
                            f"\t[bold orange_red1]\\[Meteor] {round((metric_sum_dict['meteor'] / idx) * 100, 2)}[/bold orange_red1]"
                            f"\t[bold orange_red1]\\[ExactMatch] {round((metric_sum_dict['exact-match'] / idx) * 100, 2)}[/bold orange_red1]"
                        )
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
                if args.verbal:
                    try:
                        console.print(
                            f"[bold dark_olive_green3]\[Synthesis]\n- {response_display}[/bold dark_olive_green3]\n[bold purple]\[Reference]\n- {instance['word']}: {context_display}[/bold purple]"
                        )
                    except:
                        console.print(
                            f"[bold dark_olive_green3]\[Pred] {instance['word']}: [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Ref] [underline] {instance['context']}[/underline][/bold purple]"
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
                        console.print(
                            f"[bold dark_olive_green3]\[Synthesis]\n- {response_display}[/bold dark_olive_green3]\n[bold purple]\[Reference]\n- {definition_display}[/bold purple]"
                        )
                    except:
                        console.print(
                            f"[bold dark_olive_green3]\[Pred] {instance['word']}: [underline]{response}[/underline][/bold dark_olive_green3]\t[bold purple]\[Ref] [underline] {instance['definition']}[/underline][/bold purple]"
                        )
            if task == "word-sense-disambiguation":
                if args.evaluate:
                    metric_dict = {
                        "exact-match": 0.0,
                    }
                    for k in metric_sum_dict.keys():
                        metric_dict[k] = compute_metric(
                            metric_name=k,
                            pred=response,
                            gold=instance["label"],
                        )
                        metric_sum_dict[k] += metric_dict[k]
                    if args.verbal:
                        console.print(
                            f"[bold dark_olive_green3]\\[Pred] {instance['word']}: [underline]{response}[/underline][/bold dark_olive_green3]"
                            f"\t[bold purple]\\[Gold] {instance['label']} [/bold purple]"
                            f"\t[bold orange_red1]\\[ExactMatch] {round((metric_sum_dict['exact-match'] / idx) * 100, 2)}[/bold orange_red1]"
                        )
                else:
                    if args.verbal:
                        console.print(
                            f"[bold dark_olive_green3]\\[Pred] {instance['word']}: [underline]{response}[/underline][/bold dark_olive_green3]"
                            f"\t[bold purple]\\[Gold] {instance['label']} [/bold purple]"
                        )

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
            # for k in metric_sum_dict.keys():
            # instance[k] = metric_dict[k]
            json_list_output.append(instance)

            # dump data
            if "json" in args.output_format:
                output_path = f"results/{args.task}-{args.shot_num}-shot-{args.model.replace('/', '-')}.json".replace("--", "-")
                dump_json(output_path, json_list_output, indent=True)
            elif "tsv" in args.output_format:
                output_path = f"results/{args.task}-{args.shot_num}-shot-{args.model.replace('/', '-')}.tsv".replace("--", "-")
                dump_tsv(output_path, json_list_output)

        console.rule(title="[bold yellow]End Running[/bold yellow]")

    console.print()
    console.rule(title="[bold yellow]Running Report[/bold yellow]")
    # TODO: add performance reporter
    table = Table(
        title="Running Reporter", show_header=True, header_style="bold magenta"
    )
    table.add_column("Model")
    table.add_column("Task")
    table.add_column("Shot Num")
    for k in metric_sum_dict.keys():
        table.add_column(k)
    table.add_row(
        args.model,
        task,
        str(args.shot_num),
        *[
            str(round(metric_sum_dict[k] / idx, 3) * 100)
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
        model_type = "sparse" if "xLlama" in args.model else "dense"
        model, tokenizer = init_local_model(args.model, console, model_type=model_type)

    request_llm(args, model, tokenizer)  # request GPT-3.5-turbo / GPT-4
