#对于所有缓存结果，重新计算所有指标 ✅
from eval import compute_metric, get_word2refs
from calculate_scores import compute_mauve
import argparse
import json
import re
import math
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


def get_inference_results(data_path):
    examples = []
    with open(data_path, "r") as file:
        content = file.read()
    
    json_objects = re.findall(r'\{.*?\}', content, re.DOTALL)

    for json_str in json_objects:
        datapoint = json.loads(json_str)

        examples.append({
            "word": datapoint["word"],
            "source": datapoint["source"],
            "context": datapoint["context"],
            "instruction": datapoint["instruction"],
            "definition": datapoint["definition"],
            "prediction": datapoint["prediction"]   
        }
        )

    words = [example["word"] for example in examples]
    definitions = [example["definition"] for example in examples]
    predictions = [example["prediction"] for example in examples]
    word2refs = get_word2refs(words, definitions, dedup_refs=False)

    return examples, words, definitions, predictions, word2refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_file_path", type=str, required=True, help="The path to inference result")
    args = parser.parse_args()
    
    inference_res_path = args.res_file_path
    instances, _, golds, preds, word2refs = get_inference_results(inference_res_path)


    metric_sum_dict = {
                        "bleu-cpp": 0.0,  # 1.
                        "rouge": 0.0,  # 2.
                        "meteor": 0.0,  # 3.
                        "bert-score": 0.0,  # 4.
                        "mover-score": 0.0,  # 5.
                        #"mauve": 0.0,  # 6.
                        }

    metric_dict = {
                            "bleu-cpp": 0.0,  # 1.
                            "rouge": 0.0,  # 2.
                            "meteor": 0.0,  # 3.
                            "bert-score": 0.0,  # 4.
                            "mover-score": 0.0,  # 5.
                            #"mauve": 0.0,  # 6.
                        }
    
    print(f"Start calculating metrics on {inference_res_path}...")

    # recalculate mauve using two distributions
    mauve_score = compute_mauve(preds, golds)
    print("MAUVE:", mauve_score)

    # other metrics
    for i, instance in enumerate(instances):
        for k in metric_sum_dict.keys():
            score = compute_metric(
                metric_name=k,
                pred=instance["prediction"],
                gold=(
                    instance["references"]
                    if "references" in instance
                    else instance["definition"]
                ),
                word=instance["word"],
                refs=word2refs[instance["word"]],
            )
            metric_dict[k] = (
                score
                if isinstance(score, float) or isinstance(score, int)
                else max(score)
            )
            #print(f"{k}: {metric_dict[k]}")
            metric_sum_dict[k] += metric_dict[k]
        print(f"Query {i} completed: {metric_sum_dict}")
    
    print(f"End calculating all metrics on {inference_res_path} ...")

    print("Start averaging metrics...")
    for k in metric_sum_dict.keys():
        if "mauve" not in k:
            metric_sum_dict[k] = round(metric_sum_dict[k] / len(instances), 6)
        else:
            metric_sum_dict["mauve"] = round(mauve_score, 6)


    print(inference_res_path, metric_sum_dict)
    

if __name__ == "__main__":
    main()