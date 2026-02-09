# -*- coding: utf-8 -*-
#
# @author: Yang Liu <yangliu.real@gmail.com>
# @date: 2024/04/03
#
# Preprocess data to the instruction tuning format from the original 3D-EX dataset

import os
import json
import random
from ast import literal_eval
from typing import Any, List, Dict, Tuple

import pandas as pd
from tqdm import tqdm


def prepare_instruction_tuning_data(
    input_fpath: str, output_path: str, do_split: bool = False, do_demo: bool = False
) -> None:
    def flatten_nested_list(lst: List[Any]) -> List[Any]:
        return [item for row in lst for item in row]

    def get_statistics(word2instance) -> Tuple[int, int]:
        num_examples = 0
        num_definitions = 0
        for word, instances in word2instance.items():
            num_definitions += len(instances)
            for instance in instances:
                num_examples += len(instance["examples"])
        return num_definitions, num_examples

    def preprocess(
        single_example: bool = False, startwith_alpha: bool = False
    ) -> Dict[str, List[Dict[str, Any]]]:
        df = pd.read_csv(input_fpath, low_memory=False)
        word2instance = dict()
        with open(input_fpath, "r") as f:
            for i, line in tqdm(df.iterrows()):
                word = line["TERM"]
                definition = line["DEFINITION"]
                # judge if line['EXAMPLE'] is nan, skip
                if (
                    isinstance(word, float)
                    or isinstance(definition, float)
                    or pd.isnull(line["EXAMPLE"])
                    or pd.isnull(line["DATASET_NAME"])
                    or any(
                        [
                            d in line["DATASET_NAME"]
                            for d in ["Hei++", "MultiRD", "Webster's Unabridged"]
                        ]
                    )
                ):
                    continue

                if startwith_alpha and not word[0].isalpha():
                    continue

                # map multiple contexts to the same term
                # examples = [
                # e
                # for e in flatten_nested_list(literal_eval(line["EXAMPLE"]))
                # if not isinstance(e, float)
                # ]
                examples_list = literal_eval(line["EXAMPLE"])
                # dataset_names = [
                # e
                # for e in flatten_nested_list(literal_eval(line["DATASET_NAME"]))
                # if not isinstance(e, float)
                # ]
                dataset_name_list = [
                    literal_eval(name_str)
                    for name_str in literal_eval(line["DATASET_NAME"])
                ]
                # if single_example:
                # examples = [examples[0]]
                # dataset_names = [dataset_names[0]]
                if word not in word2instance:
                    word2instance[word] = []
                # example = random.choice(examples)
                # FIXME: bug here
                for examples, dataset_name in zip(examples_list, dataset_name_list):
                    for example in examples:
                        if not isinstance(example, str) or not example:
                            continue
                        definition_norm = " ".join(definition.split())
                        example_norm = " ".join(example.split())
                        word2instance[word].append(
                            {
                                "term": word,
                                "definition": definition_norm,
                                "context": example_norm,
                                "instruction": instruction.replace(
                                    "{{word}}", word
                                ).replace("{{context}}", example_norm),
                                "source": dataset_name[0],
                            }
                        )
        # print(get_statistics(word2instance))
        return word2instance

    def train_test_split(
        word2instance: Dict[str, Any], test_size: float, random_state: int
    ) -> Tuple[Dict[str, List[Dict[str, Any]]]]:
        if test_size == 0.1:
            factor = 10
        elif test_size == 0.2:
            factor = 5
        elif test_size == 0.5:
            factor = 2
        else:
            raise ValueError("test_size must be one of 0.1, 0.2, 0.5")
        train_word2instance = dict()
        valid_test_word2instance = dict()
        # shuffle word2instance
        word2instance_list = list(word2instance.items())
        random.shuffle(word2instance_list)
        word2instance = dict(word2instance_list)
        for idx, (word, instances) in enumerate(word2instance.items(), start=1):
            if idx % factor == 0:
                valid_test_word2instance[word] = instances
            else:
                train_word2instance[word] = instances
        return train_word2instance, valid_test_word2instance

    # instruction = "Assume that you are a linguistic who research in the word definition. You will be given a word associated with a given context containing the word. Your task is to give the specific properly definition of the word in the context.\n\nWord: {{word}}\n\nContext: {{context}}\n\nWord definition:"
    instruction = '"{{context}}" What is the definition of "{{word}}"?'
    word2instance = preprocess()

    if do_demo:
        raise NotImplementedError("Demo is not implemented yet")

    if do_split:
        train_word2instance, valid_test_word2instance = train_test_split(
            word2instance, test_size=0.2, random_state=42
        )
        valid_word2instance, test_word2instance = train_test_split(
            valid_test_word2instance, test_size=0.5, random_state=42
        )

        with open(f"{output_path}/train.jsonl", "w") as f:
            for word, instances in train_word2instance.items():
                for instance in instances:
                    f.write(json.dumps(instance) + "\n")

        with open(f"{output_path}/valid.jsonl", "w") as f:
            for word, instances in valid_word2instance.items():
                for instance in instances:
                    f.write(json.dumps(instance) + "\n")

        with open(f"{output_path}/test.jsonl", "w") as f:
            for word, instances in test_word2instance.items():
                for instance in instances:
                    f.write(json.dumps(instance) + "\n")
    else:
        with open(output_path, "w") as f:
            for word, instances in word2instance.items():
                for instance in instances:
                    f.write(json.dumps(instance, ensure_ascii=False) + "\n")


def prepare_instruction_tuning_data_naacl(
    input_dir: str,
    output_dir: str,
    do_demo: bool = False,
    dedup_training: bool = False,
    up_sampling: bool = False,
    add_corruption_settings: bool = False,
) -> None:
    """
    Prepare the instruction tuning data splits (train/valid/test) from the NAACL dataset:
    https://github.com/shonosuke/ishiwatari-naacl2019
    Steps for reproducing the dataset:
    1. `wget http://www.tkl.iis.u-tokyo.ac.jp/~ishiwatari/naacl_data.zip`
    2. `unzip naacl_data.zip`

    Args:
    - input_dir: the directory containing the NAACL dataset
    - output_dir: the directory to save the instruction tuning data splits
    - do_demo: whether to run the demo or not
    - dedup_training: whether to deduplicate the training data or not
    - up_sampling: whether to up-sample the training data or not
    - add_corruption_settings: whether to add corruption settings or not

    Returns:
    - None
    """
    os.makedirs(output_dir, exist_ok=True)
    train_fpath_eg = f"{input_dir}/train.eg"
    train_fpath_txt = f"{input_dir}/train.txt"

    valid_fpath_eg = f"{input_dir}/valid.eg"
    valid_fpath_txt = f"{input_dir}/valid.txt"

    test_fpath_eg = f"{input_dir}/test.eg"
    test_fpath_txt = f"{input_dir}/test.txt"

    train_data, valid_data, test_data = [], [], []
    # instruction_template = '"{{context}}" What is the definition of "{{word}}"?'
    instruction_template = "[Term] {{word}}\n[Context] {{context}}\n[Definition] "
    # instruction_template = "{{word}}\n{{context}}\n"
    for fpath_eg, fpath_txt, data in [
        (train_fpath_eg, train_fpath_txt, train_data),
        (valid_fpath_eg, valid_fpath_txt, valid_data),
        (test_fpath_eg, test_fpath_txt, test_data),
    ]:
        with open(fpath_eg, "r") as f_eg, open(fpath_txt, "r") as f_txt:
            for line_eg, line_txt in zip(f_eg, f_txt):
                items_eg = line_eg.strip().split("\t")
                items_txt = line_txt.strip().split("\t")
                word = items_eg[0].split("%", maxsplit=1)[0].strip()
                context = items_eg[1].replace("<TRG>", word).strip()
                pos = items_txt[1].strip()
                source = items_txt[2].strip()
                reference = items_txt[3].strip()
                if word not in context:
                    print(
                        f"Drop example:\tword: {word}, context: {context}, source: {source}"
                    )
                    continue
                instruction = instruction_template.replace(
                    "{{context}}", context
                ).replace("{{word}}", word)

                # TODO: verify the content of the two synonyms column
                synonyms_1 = [
                    item.split("%", maxsplit=1)[0].strip()
                    for item in (
                        items_txt[4]
                        .strip()
                        .split("[", maxsplit=1)[-1]
                        .rsplit("]", maxsplit=1)[0]
                        .split(" ")
                    )
                ]
                # TODO: verify the content of the two synonyms column
                synonyms_2 = [
                    item.split("%", maxsplit=1)[0].strip()
                    for item in (
                        items_txt[5]
                        .strip()
                        .split("[", maxsplit=1)[-1]
                        .rsplit("]", maxsplit=1)[0]
                        .split(" ")
                    )
                ]

                synonyms = list(set(synonyms_1 + synonyms_2))
                if synonyms[0] == "":
                    synonyms = []

                data.append(
                    {
                        "term": word,
                        "pos": pos,
                        "context": context,
                        "definition": reference,
                        "source": source,
                        "synonyms": synonyms,
                        "instruction": instruction,
                    }
                )

    if do_demo:
        raise NotImplementedError("Demo is not implemented yet")

    with open(f"{output_dir}/train.structure.jsonl", "w") as f:
        for instance in train_data:
            f.write(json.dumps(instance) + "\n")

    with open(f"{output_dir}/valid.structure.jsonl", "w") as f:
        for instance in valid_data:
            f.write(json.dumps(instance) + "\n")

    with open(f"{output_dir}/test.structure.jsonl", "w") as f:
        for instance in test_data:
            f.write(json.dumps(instance) + "\n")

    # Add corruption settings and save the data
    if add_corruption_settings:
        for data_split, data in [
            ("train", train_data),
            ("valid", valid_data),
            ("test", test_data),
        ]:
            with open(f"{output_dir}/{data_split}_corrupted.jsonl", "w") as f:
                for instance in data:
                    corrupted_instance = instance.copy()
                    corrupted_instance["definition"] = "corrupted"
                    f.write(json.dumps(corrupted_instance) + "\n")

    # Whether do deduplication and up-sampling for training data or not
    if dedup_training:
        # Deduplicate training data by term with specific seed
        for seed in [21, 42, 84]:
            os.makedirs(f"{output_dir}/train_dedup", exist_ok=True)
            with open(f"{output_dir}/train_dedup/train_dedup_{seed}.jsonl", "w") as f:
                random.seed(seed)
                random.shuffle(train_data)
                train_data_dedup = list(
                    {instance["term"]: instance for instance in train_data}.values()
                )
                if up_sampling:
                    target_training_data_size = len(train_data)
                    train_data_dedup_upsampling = train_data_dedup * (
                        (target_training_data_size // len(train_data_dedup)) + 1
                    )
                    train_data_dedup = train_data_dedup_upsampling[
                        :target_training_data_size
                    ]
                    random.shuffle(train_data_dedup)
                for instance in train_data_dedup:
                    f.write(json.dumps(instance) + "\n")


def convert_to_nemo_data_format(input_fpath: str, output_fpath: str) -> None:
    data = []
    with open(input_fpath, "r") as f:
        for line in f:
            instance = json.loads(line)
            data.append(
                {
                    "input": instance["instruction"],
                    "output": instance["definition"],
                }
            )

    with open(output_fpath, "w") as f:
        for instance in data:
            f.write(json.dumps(instance, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # for dataset in ["wordnet", "oxford", "wiki", "slang", "3D-EX"]:
    # for split in ["train", "valid", "test"]:
    # convert_to_nemo_data_format(
    # input_fpath=f"dataset/{dataset}/{split}.jsonl",
    # output_fpath=f"dataset/{dataset}/{split}.nemo.jsonl",
    # )
    # prepare_instruction_tuning_data(
    # input_fpath=f"/home/ivanfung/workspace/LM-Lexicon/data/lexical_{split}_term.csv",
    # output_path=f"dataset/3D-EX/{split}.jsonl",
    # )
    for dataset in ["wordnet", "oxford", "wiki", "slang"]:
        prepare_instruction_tuning_data_naacl(
            input_dir=f"dataset/naacl_data/{dataset}",
            output_dir=f"dataset/{dataset}",
            do_demo=False,
            dedup_training=False,
            up_sampling=False,
        )
