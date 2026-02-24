# -*- coding: utf-8 -*-
#
# @Author: YangLiu <yangliu.real@gmail.com>
# @Date: 2023/10/31

import re
import ast
import json
import random
from argparse import Namespace
from typing import List, Dict, Any, Optional

from rich import print_json
from openicl import (
    DatasetReader,
    TopkRetriever,
    PromptTemplate,
)
from datasets import load_dataset

from type import LF_CATEGORY_ID2LABEL_MAP_8, LF_CATEGORY_LIST_8


random.seed(42)


def get_lexical_data(
    data_path: str, task_type: str, max_num_limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get lexical data from data path

    Args:
    - data_path: data path
    - task_type: task type
    - max_num_limit: max number of data

    Returns:
    - data examples
    """
    examples = []
    with open(data_path, "r") as f:
        for idx, line in enumerate([l.strip() for l in f.readlines()]):
            if len(examples) >= max_num_limit:
                break
            datapoint = json.loads(line)
            examples.append(
                {
                    # "id": idx,
                    "word": datapoint["term"],
                    "context": datapoint["context"],
                    "definition": datapoint["definition"],
                    "source": datapoint["source"] if "source" in datapoint else "3D-EX",
                    "instruction": datapoint["instruction"],
                }
            )

    random.shuffle(examples)

    return examples


def get_wsd_data(
    data_path: str, task_type: str, max_num_limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get lexical data from data path

    Args:
    - data_path: data path
    - task_type: task type
    - max_num_limit: max number of data

    Returns:
    - data examples
    """
    examples = []
    with open(data_path, "r") as f:
        for idx, line in enumerate([l.strip() for l in f.readlines()]):
            if len(examples) >= max_num_limit:
                break
            datapoint = json.loads(line)
            examples.append(
                {
                    "word": datapoint["word"],
                    "context": datapoint["input_text"],
                    "options": datapoint["options"],
                    "label": datapoint["label"],
                    "instruction": datapoint["instruction"],
                }
            )

    random.shuffle(examples)

    return examples


def prepare_dense_retriever(
    examples: List[Dict[str, Any]], top_k: int
) -> TopkRetriever:
    """
    Prepare open in-context learning retriever

    Args:
    - examples: examples
    - top_k: top k examples to retrieve

    Returns:
    - retriever
    """
    # Loading dataset from huggingface
    dataset = load_dataset("gpt3mix/sst2")
    # Define a DatasetReader, with specified column names where input and output are stored.
    data = DatasetReader(dataset, input_columns=["text"], output_column="label")

    # Define the prompt template (Optional)
    tp_dict = {
        0: "</E>Positive Movie Review: </text>",
        1: "</E>Negative Movie Review: </text>",
    }
    template = PromptTemplate(tp_dict, {"text": "</text>"}, ice_token="</E>")

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = TopkRetriever(data, ice_num=top_k)

    return retriever


def load_icl_example(
    data_path: str,
    task_type: str,
    max_num_limit: int,
    retrieval_policy: str = "random",
) -> List[Dict[str, Any]]:
    """
    Load in-context learning data example from data path

    Args:
    - data_path: data path
    - task_type: task type
    - max_num_limit: max number of data
    - retrieval_policy: specific policy for examples retrieval

    Returns:
    - data examples
    """
    data_examples = []
    if task_type == "word-interpretation":
        with open(data_path, "r", encoding="utf8") as f:
            examples = [json.loads(line.strip()) for line in f.readlines()]
            if max_num_limit > len(examples):
                raise ValueError("shot number limit exceeded")
        if retrieval_policy == "random":
            data_examples = random.choices(examples, k=max_num_limit)
        elif retrieval_policy == "sequencial":
            data_examples = examples[:max_num_limit]
        elif retrieval_policy == "open-icl":
            # TODO: retriever = prepare_dense_retriever(examples, max_num_limit)
            # data_examples = retriever.retrieve(max_num_limit)
            raise NotImplementedError("Not implemented yet!")
    else:
        raise NotImplementedError(
            "Only support following list of tasks for now: [word-interpretation]"
        )

    return data_examples


def construct_prompt(
    args: Namespace,
    instance: Dict[str, Any],
    prompt_template: str,
    examples: Optional[List[Dict[str, Any]]] = None,
    is_oracle: Optional[bool] = None,
) -> str:
    """
    Construct prompt from the given instance, prompt template, taxonomy, and examples

    Args:
    - args: arguments
    - instance: instance
    - prompt_template: prompt template
    - taxonomy: taxonomy
    - examples: examples

    Returns:
    - constructed prompt
    """
    prompt = ""
    # prepare prompt
    if args.task in [
        "word-interpretation",
        "synthesize-context",
        "synthesize-definition",
    ]:
        prompt = prompt_template.replace("{{word}}", instance["word"])
        prompt = prompt.replace("{{context}}", instance["context"])
        prompt = prompt.replace("{{definition}}", instance["definition"])
    if args.task == "word-sense-disambiguation":
        prompt = prompt_template.replace("{{context}}", instance["context"])
    if examples and "{{example}}" in prompt_template:
        if args.task == "word-interpretation":
            if "gpt" in args.model.lower():
                example_str = "".join(
                    [
                        f"\n\"{example['context']}\" What is the definition of \"{example['term']}\"? {example['definition']}\n"
                        for example in examples[: args.shot_num]
                    ]
                )
            else:
                example_str = "".join(
                    [
                        f"\n\"{example['context']}\" What is the definition of \"{example['term']}\"? {example['definition']}\n"
                        for example in examples[: args.shot_num]
                    ]
                )
            prompt = (
                prompt_template.replace(
                    "{{example}}",
                    example_str,
                )
                if prompt == ""
                else prompt.replace(
                    "{{example}}",
                    example_str,
                )
            )
        else:
            raise NotImplementedError("Not implemented yet!")

    if args.shot_num == 0:
        return prompt.replace("{{example}}\n", "")

    return prompt


def postprocess(
    s: str, task: Optional[str] = None, args: Optional[Namespace] = None
) -> str:
    if task in ["synthesize-context", "synthesize-definition"]:
        s = s.replace(" ### ", "###").replace(" ###", "###").replace("### ", "###")
        s = (
            s.replace("1. ", "")
            .replace("1.", "")
            .replace("2. ", "")
            .replace("2.", "")
            .replace("3. ", "")
            .replace("3.", "")
            .replace("4. ", "")
            .replace("4.", "")
            .replace("5. ", "")
            .replace("5.", "")
            .replace("6. ", "")
            .replace("6.", "")
            .replace("7. ", "")
            .replace("7.", "")
            .replace("8. ", "")
            .replace("8.", "")
            .replace("9. ", "")
            .replace("9.", "")
            .replace("10. ", "")
            .replace("10.", "")
        )
    if task == "word-interpretation":
        if "claude" in args.model.lower():
            if "I apologize" in s:
                return ""
            s = s.rsplit('"?')[-1]
            if ":" in s:
                s = s.split(":")[-1]
            if "means" in s:
                s = s.split("means")[-1]
    return s.strip()
