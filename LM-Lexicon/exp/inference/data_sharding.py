import os
import json
import random
from time import sleep
from typing import Dict, List

import httpx
import openai
import numpy as np
from tqdm import tqdm
from rich.console import Console
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util

console = Console()
dataset = "3D-EX"  # "oxford", "wiki", "slang", "3D-EX"
sample_num = 10000000
split = "train"
# model_is_ok = False

model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
# while not model_is_ok:
    # try:
        # model = SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
    # except:
        # sleep(60)
        # continue
    # else:
        # model_is_ok = True

embedding_fpath = f"/data2/checkpoints/data/{dataset}.{split}.json"

cluster_centroids = 4

httpx_client = httpx.Client(
    proxies={"https://": "http://127.0.0.1:7895", "http://": "http://127.0.0.1:7895"}
)

# client = openai.Client(
# api_key="sk-Wb6cjFemPM9UHDE4F68bB69f8cA643698cC9Ac27AbD54dF9",
# base_url="https://ai56.top/v1",
# http_client=httpx_client,
# )


def get_random_example_word_definition(
    dataset: str = "3D-EX", max_num_limit: int = 1000
) -> List[Dict[str, str]]:
    examples = []
    with open("dataset/3D-EX/train.jsonl") as f:
        datapoints = [json.loads(l.strip()) for l in f.readlines()]
        if max_num_limit < len(datapoints):
            datapoints = random.choices(datapoints, k=max_num_limit)
        for datapoint in datapoints:
            term = datapoint["term"]
            context = datapoint["context"]
            definition = datapoint["definition"]
            instruction = datapoint["instruction"]
            examples.append(
                {
                    "term": term,
                    "context": context,
                    "definition": definition,
                    "instruction": instruction,
                }
            )
    return examples


examples = get_random_example_word_definition(dataset, sample_num)

console.log(examples[:3])

if os.path.isfile(embedding_fpath):
    print("Loading existed file:", embedding_fpath)
    with open(embedding_fpath, "r") as f:
        examples = json.loads(f.read().strip())
else:
    print("Retrieving embeddings from Model ...")
    for example in tqdm(examples):
        """
        # use OpenAI "text-embedding-3-large" Model
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=[example["term"] + "; " + example["definition"]],  # [example["term"] + example["definition"] + example["context"]]
        )
        example["embedding"] = response.data[0].embedding
        """
        # use our locally deployed "SFR-Embedding-Mistral" model
        text = example["definition"]
        example["embedding"] = model.encode(text).tolist()

    with open(embedding_fpath, "w") as f:
        f.write(json.dumps(examples, ensure_ascii=False))

embeddings = np.array([example["embedding"] for example in examples])

# Cluster dataset to several clusters by trained k-means clusterer
kmeans_clusterer = KMeans(
    n_clusters=cluster_centroids,
    algorithm="elkan",
    init="k-means++",
    n_init=10,
    max_iter=1000,
    tol=1e-8,
    random_state=42,
)
kmeans_labels = kmeans_clusterer.fit_predict(embeddings)

cluster2examples = dict()
for cluster in range(cluster_centroids):
    print(f"Cluster {cluster}")
    for example in examples:
        if kmeans_clusterer.predict([example["embedding"]])[0] == cluster:
            if cluster not in cluster2examples:
                cluster2examples[cluster] = []
            cluster2examples[cluster].append(example.copy())
    # retrieve top-100 examples to print
    for example in cluster2examples[cluster][:100]:
        console.log(f" - {example['term']}: {example['definition']}")
    print()

os.makedirs("dataset/clustered", exist_ok=True)
for cluster, examples in cluster2examples.items():
    os.makedirs(f"dataset/clustered/cluster-{str(cluster)}", exist_ok=True)
    with open(f"dataset/clustered/cluster-{str(cluster)}/train.jsonl", "w") as f1, \
            open(f"dataset/clustered/cluster-{str(cluster)}/train.indent.jsonl", "w") as f2:
        for example in examples:
            example.pop("embedding")
            f1.write(json.dumps(example, ensure_ascii=False) + "\n")
            f2.write(json.dumps(example, indent=4, ensure_ascii=False) + "\n")
