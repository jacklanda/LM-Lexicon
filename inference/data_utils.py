# -*- coding: utf-8 -*-
#
# @Author: YangLiu <yangliu.real@gmail.com>
# @Date: 2023/10/31

import re
import os
import ast
import json
import random
from argparse import Namespace
from typing import List, Dict, Any, Optional, Tuple

from rich import print_json
from rich.console import Console
# from openicl import (
    # DatasetReader,
    # TopkRetriever,
    # BM25Retriever,
    # VotekRetriever,
    # PromptTemplate,
# )
from datasets import load_dataset

from type import LF_CATEGORY_ID2LABEL_MAP_8, LF_CATEGORY_LIST_8

# random.seed(42)
random.seed(128)

console = Console()


def save_selected_data(
    data_points: List[Dict[str, Any]], data_path: str, max_num_limit: int
):
    """
    Save selected data points to a new jsonl file.

    Args:
    - data_points: list of selected data points
    - data_path: original data path
    - max_num_limit: max number of data points
    """
    directory, filename = os.path.split(data_path)
    filename = os.path.splitext(filename)[0]

    new_filename = f"{filename}_{max_num_limit}.jsonl"
    new_filepath = os.path.join(directory, new_filename)

    with open(new_filepath, "w") as f:
        for datapoint in data_points:
            f.write(json.dumps(datapoint) + "\n")

    # print(f"Selected data saved to {new_filepath}")


def get_lexical_data(
    data_path: str,
    task_type: str,
    max_num_limit: Optional[int] = None,
    seed: Optional[int] = 42,
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, List[str]]]:
    """
    Get lexical data from data path

    Args:
    - data_path: data path
    - task_type: task type
    - max_num_limit: max number of data
    - seed: random seed for reproducibility

    Returns:
    - data examples, words list, and word-to-definitions mapping
    """
    with open(data_path, "r") as f:
        data_points = [json.loads(line.strip()) for line in f.readlines()]

    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Randomly select max_num_limit data points
    if max_num_limit is not None and max_num_limit < len(data_points):
        selected_data_points = random.sample(data_points, max_num_limit)
    else:
        selected_data_points = data_points

    # Save selected data points to a new jsonl file
    # save_selected_data(selected_data_points, data_path, max_num_limit)

    examples = [
        {
            "word": datapoint["term"],
            "context": datapoint["context"].replace("\n", ""),
            "definition": datapoint["definition"].replace("\n", ""),
            "source": datapoint["source"].replace("\n", "")
            if "source" in datapoint
            else "3D-EX",
            "instruction": datapoint["instruction"].replace("\n", ""),
        }
        for datapoint in selected_data_points
    ]

    # dedup by "word" + "context"
    # """
    examples_dedup = []
    seen = set()
    for example in examples:
        # key = example["word"] + example["context"]
        key = example["word"]
        if key not in seen:
            examples_dedup.append(example)
            seen.add(key)
    examples = examples_dedup
    # """

    console.log(f"Total number of examples: {len(examples)}")
    random.shuffle(examples)

    words = [example["word"] for example in examples]
    refs = [example["definition"] for example in examples]
    data_dir = data_path.split("/")[1]
    data_path = f"dataset/{data_dir}/word2refs.json"
    # if os.path.exists(data_path):
        # with open(data_path, "r") as f:
            # word2refs = json.load(f)
    # else:
    console.log("Creating word2refs mapping...")
    word2refs = get_word2refs(words, refs, dedup_refs=False)
    with open(data_path, "w") as f:
        json.dump(word2refs, f, ensure_ascii=False, indent=4)

    return examples, words, word2refs


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
    data_path: str, top_k: int, max_num_limit: Optional[int] = 10
):
    """
    Prepare open in-context learning retriever

    Args:
    - data_path: path to source data file
    - top_k: top k examples to retrieve

    Returns:
    - retriever
    """
    # Loading dataset from huggingface
    train_file_path = os.path.join(data_path, "train.jsonl")
    test_file_path = os.path.join(data_path, f"test_{max_num_limit}.jsonl")

    data_files = {
        "train": train_file_path,
        "test": test_file_path,
    }

    dataset = load_dataset("json", data_files=data_files)
    # Define a DatasetReader, with specified column names where input and output are stored.
    data = DatasetReader(
        dataset, input_columns=["term", "context"], output_column="definition"
    )

    # Define the prompt template (Optional)
    tp_dict = {
        0: "</E>Positive Movie Review: </text>",
        1: "</E>Negative Movie Review: </text>",
    }
    template = PromptTemplate(tp_dict, {"text": "</text>"}, ice_token="</E>")

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = TopkRetriever(
        data,
        ice_num=top_k,
        index_split="train",
        test_split="test",
        batch_size=128,
    )  # other paras: tokenizer, batchsize

    return retriever


def load_icl_example(
    data_path: str,
    task_type: str,
    shot_num: int,
    max_num_limit: int,
    retrieval_policy: str,
) -> List[Dict[str, Any]]:
    """
    Load in-context learning data example from data path

    Args:
    - data_path: data path
    - task_type: task type
    - shot_num: number of examples to retrieve
    - max_num_limit: max number of datapoints
    - retrieval_policy: specific policy for examples retrieval

    Returns:
    - data examples
    """
    data_examples = []
    if task_type == "word-interpretation":
        # train_file_path = os.path.join(data_path, "train.jsonl")
        train_file_path = data_path
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(f"Can't find the file: {train_file_path}")

        with open(train_file_path, "r", encoding="utf8") as f:
            examples = [json.loads(line.strip()) for line in f.readlines()]
            if shot_num > len(examples):
                raise ValueError("shot number limit exceeded")
        if retrieval_policy == "random":
            data_examples = random.choices(examples, k=shot_num)
            # print(data_examples)
        elif retrieval_policy == "sequencial":
            data_examples = examples[:shot_num]
        elif retrieval_policy == "topk":
            retriever = prepare_dense_retriever(
                data_path, top_k=shot_num, max_num_limit=max_num_limit
            )
            data_examples_list = retriever.retrieve()
            # fetch training examples according to the near idx
            data_examples = [
                [examples[idx] for idx in idx_list] for idx_list in data_examples_list
            ]
            # print(data_examples)

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
                        f"\n\"{example['context']}\" What is the definition of \"{example['term']}\"? {example['definition']}\n"  # style 0
                        # f"\n[QUESTION] What is the definition of \"{example['term']}\" in the context \"{example['context']}\"? [CONCISE DEFINITION] {example['definition']}\n"  # style 1
                        # f"\n[QUESTION] What is the definition of \"{example['term']}\" in the context? [CONTEXT] {example['context']} [CONCISE DEFINITION] (short and concise) {example['definition']}\n"  # style 2
                        # f"\n[CONTEXT] {example['context']}\n[QUESTION] What is the definition of \"{example['term']}\" in the context?\n[CONCISE DEFINITION] (short and concise) {example['definition']}\n"  # style 3
                        for example in examples[: args.shot_num]
                    ]
                )
            else:
                example_str = "".join(
                    [
                        f"\n\"{example['context']}\" What is the definition of \"{example['term']}\"? {example['definition']}\n"  # style 0
                        # f"\n[QUESTION] What is the definition of \"{example['term']}\" in the context \"{example['context']}\"? [CONCISE DEFINITION] {example['definition']}\n"  # style 1
                        # f"\n[QUESTION] What is the definition of \"{example['term']}\" in the context? [CONTEXT] {example['context']} [CONCISE DEFINITION] (short and concise) {example['definition']}\n"  # style 2
                        # f"\n[CONTEXT] {example['context']}\n[QUESTION] What is the definition of \"{example['term']}\" in the context?\n[CONCISE DEFINITION] (short and concise) {example['definition']}\n"  # style 3
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
            if s.count("\n\n") > 1:
                chunks = s.split("\n\n")
                s = chunks[-2].strip() + "\n\n" + chunks[-1].strip()
            elif s.count("\n\n") == 1:
                s = s.split("\n\n")[-1].strip()
            else:
                s = s.strip()
            # if s is not None and "I apologize" in s:
            # return ""
            # s = s.rsplit('"?')[-1]
            # if s is not None and ":" in s:
            # s = s.split(":")[-1]
            # if s is not None and "means" in s:
            # s = s.split("means")[-1]
            # s = s.split("\n")[0]
        if "gpt" in args.model.lower():
            if ":" in s:
                s = s.split(":")[-1]
            if "means" in s:
                s = s.split("means")[-1]
            if "refers to" in s:
                s = s.split("refers to")[-1]
            if " is " in s:
                pattern = r'The definition of "([^"]*)" is'
                s = re.sub(pattern, "", s)
            s = s.replace("(short and concise)", "")
            try:
                pattern = r"\*\*(.*?)\*\*"
                s = re.search(pattern, s).group(1)
            except Exception as _:
                pass
            s = s.strip("**")
            if s.startswith("****"):
                s = s[4:]
            if s.startswith("**"):
                s = s[2:]
            if s.startswith("*"):
                s = s[2:]
            if s.endswith("****"):
                s = s[:-4]
            if s.endswith("**"):
                s = s[:-2]
            if s.endswith("*"):
                s = s[:-1]
        if "gemini" in args.model.lower():
            s = s.split("\n\n")[-1].strip()
            s = s.replace("**", "")
            if ":" in s:
                s = s.split(":", 1)[1]
            if "refers to" in s:
                s = s.split("refers to")[-1]
            if "means" in s:
                s = s.split("means")[-1]
            if "Good job! 😊  Do you want to try another definition?" in s:
                s = s.replace(
                    "Good job! 😊  Do you want to try another definition?", ""
                )
            if "Let me know if you'd like to explore any other words!" in s:
                s = s.replace(
                    "Let me know if you'd like to explore any other words!", ""
                )
            if "Let me know if you'd like me to define any other words!" in s:
                s = s.replace(
                    "Let me know if you'd like me to define any other words!", ""
                )
            if "Let me know if you have any other words you'd like defined!" in s:
                s = s.replace(
                    "Let me know if you have any other words you'd like defined!", ""
                )
            if (
                "Let me know if you'd like me to elaborate on any of these definitions!"
                in s
            ):
                s = s.replace(
                    "Let me know if you'd like me to elaborate on any of these definitions!",
                    "",
                )
            if "Let me know if you need definitions for any other words! 😊 " in s:
                s = s.replace(
                    "Let me know if you need definitions for any other words! 😊 ", ""
                )
            if (
                "Do you want to continue with more word definitions? I'm ready if you are! "
                in s
            ):
                s = s.replace(
                    "Do you want to continue with more word definitions? I'm ready if you are! ",
                    "",
                )
            if "Let me know if you have any other words you'd like defined!" in s:
                s = s.replace(
                    "Let me know if you have any other words you'd like defined!", ""
                )
            if "Let me know if you'd like to tackle more definitions!" in s:
                s = s.replace(
                    "Let me know if you'd like to tackle more definitions!", ""
                )

        if "[CONCISE DEFINITION]" in s:
            s = s.split("[CONCISE DEFINITION]")[-1]

        if "(short and concise)" in s:
            s = s.split("(short and concise)")[-1]

    return s.strip() if s is not None and isinstance(s, str) else [i.strip() for i in s]


def get_word2refs(
    words: List[str],
    refs: List[str],
    dedup_refs: bool = False,
) -> Dict[str, List[str]]:
    """creates a dictionary of words to references

    Args:
        words (List[str]): list of words
        refs (List[str]): list of references
        dedup_refs (bool, optional): deduplicate references. Defaults to False.

    Returns:
        Dict[str, List[str]]: dictionary of words to references
    """
    seen_refs = set()
    word2refs = dict()
    for word, ref in zip(words, refs):
        if word not in word2refs:
            word2refs[word] = []
        if dedup_refs:
            # NOTE: deduplicating references seems to be more reasonable
            if word + ref not in seen_refs:
                seen_refs.add(word + ref)
                word2refs[word].append(ref)
        else:
            # NOTE: we follow Huang et al. (2021) to use un-deduplicated references to keep consistency
            word2refs[word].append(ref)

    return word2refs
