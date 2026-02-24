#!/usr/bin/env python

import os
import json
import random
from typing import Dict, List
from functools import partial
from collections import defaultdict
from argparse import ArgumentParser, Namespace

import numpy as np
from tqdm import tqdm
from rich.console import Console
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    HDBSCAN,
    MeanShift,
    SpectralClustering,
)
from balanced_kmeans import kmeans_equal

# from kmeans_pytorch import KMeans as KMeansPytorch
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    silhouette_score,
    pairwise_distances,
    calinski_harabasz_score,
)

console = Console()


def load_clusterer(args: Namespace):
    if args.clusterer == "kmeans":
        return KMeans(
            n_clusters=args.cluster_centroids,
            algorithm="elkan",
            init="k-means++",
            n_init=100,
            max_iter=1000,
            tol=1e-8,
            random_state=42,
        )
    elif args.clusterer == "balanced_kmeans":
        return KMeansPytorch()
        # return partial(
        # kmeans_equal,
        # num_clusters=args.cluster_centroids,
        # max_iter=1000,
        # tol=1e-8,
        # )
    elif args.clusterer == "dbscan":
        return DBSCAN(
            eps=0.5,
            min_samples=5,
            metric="euclidean",
            metric_params=None,
            algorithm="auto",
            leaf_size=30,
            p=None,
            n_jobs=None,
        )
    elif args.clusterer == "hdbscan":
        return HDBSCAN(
            min_cluster_size=args.cluster_centroids,
            metric="euclidean",
            alpha=1.0,
            min_samples=None,
            p=None,
            leaf_size=40,
            approx_min_span_tree=True,
            gen_min_span_tree=False,
            core_dist_n_jobs=4,
            cluster_selection_epsilon=0.0,
            cluster_selection_method="eom",
            allow_single_cluster=False,
            prediction_data=False,
            match_reference_implementation=False,
            optimize_memory=True,
            # memory=Memory(cachedir=None, verbose=0),
            metric_params=None,
            gen_min_span_tree_kwargs=None,
        )
    elif args.clusterer == "meanshift":
        return MeanShift(
            bandwidth=None,
            seeds=None,
            bin_seeding=False,
            min_bin_freq=1,
            cluster_all=True,
            n_jobs=None,
        )
    elif args.clusterer == "spectral":
        return SpectralClustering(
            n_clusters=args.cluster_centroids,
            eigen_solver=None,
            n_components=None,
            random_state=None,
            n_init=10,
            gamma=1.0,
            affinity="rbf",
            n_neighbors=10,
            eigen_tol=0.0,
            assign_labels="kmeans",
            degree=3,
            coef0=1,
            kernel_params=None,
            n_jobs=None,
        )
    else:
        raise NotImplementedError


def get_random_example_word_definition(
    dataset: str = "3D-EX", max_num_limit: int = -1
) -> List[Dict[str, str]]:
    examples = []
    with open(f"dataset/{dataset}/train.jsonl") as f:
        datapoints = [json.loads(line.strip()) for line in f.readlines()]
        if max_num_limit >= 0 and max_num_limit < len(datapoints):
            datapoints = random.choices(datapoints, k=max_num_limit)
        for datapoint in datapoints:
            term = datapoint["term"]
            context = datapoint["context"]
            definition = datapoint["definition"]
            instruction = datapoint["instruction"]
            source = datapoint["source"]
            examples.append(
                {
                    "term": term,
                    "context": context,
                    "definition": definition,
                    "instruction": instruction,
                    "source": source,
                }
            )
    random.shuffle(examples)
    return examples


def calculate_intercluster_distance(X, labels, cluster_centers=None) -> float:
    """
    Calculate the inter-cluster distance (distance between clusters)

    Parameters:
    X: array-like, data matrix
    labels: array-like, cluster labels
    cluster_centers: array-like, cluster centroids. If None, centroids will be calculated from the data

    Returns:
    float: average inter-cluster distance
    """
    # Get the number of clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # If there's only one cluster, inter-cluster distance is undefined
    if n_clusters <= 1:
        console.log("Only one cluster found, intercluster distance is undefined.")
        return 0.0

    # If cluster centers are not provided, calculate them
    if cluster_centers is None:
        cluster_centers = np.zeros((n_clusters, X.shape[1]))
        for i, label in enumerate(unique_labels):
            cluster_points = X[labels == label]
            if len(cluster_points) > 0:
                cluster_centers[i] = np.mean(cluster_points, axis=0)

    # Calculate distances between all cluster centers
    center_distances = pairwise_distances(cluster_centers, metric="euclidean")

    # Initialize sum of inter-cluster distances and count
    total_intercluster_distance = 0.0
    distance_count = 0

    # Calculate distances between all pairs of clusters
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # Record distance between the pair of clusters
            pair_distance = center_distances[i, j]
            total_intercluster_distance += pair_distance
            distance_count += 1

            # Output distance between each pair of clusters
            console.log(
                f"Distance between cluster {unique_labels[i]} and {unique_labels[j]}: {pair_distance:.3f}"
            )

    # Calculate average inter-cluster distance
    avg_intercluster_distance = (
        total_intercluster_distance / distance_count if distance_count > 0 else 0
    )

    return avg_intercluster_distance


def calculate_intracluster_distance(X, labels, cluster_centers=None) -> float:
    """
    Calculate the intra-cluster distance (distance within clusters)

    Parameters:
    X: array-like, data matrix
    labels: array-like, cluster labels
    cluster_centers: array-like, cluster centroids. If None, centroids will be calculated from the data

    Returns:
    float: average intra-cluster distance
    """

    # Get the number of clusters
    n_clusters = len(np.unique(labels))

    # If cluster centers are not provided, calculate them
    if cluster_centers is None:
        cluster_centers = np.zeros((n_clusters, X.shape[1]))
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                cluster_centers[i] = np.mean(cluster_points, axis=0)

    # Initialize sum of intra-cluster distances and total samples
    total_intracluster_distance = 0.0
    total_samples = 0

    # Calculate intra-cluster distance for each cluster
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        cluster_size = len(cluster_points)

        if cluster_size > 0:
            # Calculate distances from all points in the cluster to the centroid
            distances = pairwise_distances(
                cluster_points, [cluster_centers[i]], metric="euclidean"
            )
            # Sum of distances in this cluster
            cluster_distance_sum = np.sum(distances)
            # Average distance in this cluster
            cluster_avg_distance = cluster_distance_sum / cluster_size

            # Add to the total intra-cluster distance
            total_intracluster_distance += cluster_distance_sum
            total_samples += cluster_size

            # Output average intra-cluster distance for each cluster
            console.log(
                f"Cluster {i} avg intracluster distance: {cluster_avg_distance:.3f}"
            )

    # Calculate overall average intra-cluster distance
    avg_intracluster_distance = (
        total_intracluster_distance / total_samples if total_samples > 0 else 0
    )

    return avg_intracluster_distance


def split_by_lexical(args: Namespace):
    examples = get_random_example_word_definition(args.dataset, args.sample_num)

    instructions = [example["instruction"] for example in examples]
    definitions = [example["definition"] for example in examples]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)

    texts = [
        instruction + "\n\n" + definition
        for instruction, definition in zip(instructions, definitions)
    ]
    X = vectorizer.fit_transform(texts)

    clusterer = load_clusterer(args)
    labels = clusterer.fit_predict(X)

    # calculate silhouette score
    avg_silhouette_score = silhouette_score(X, labels, random_state=42)
    avg_calinski_harabasz_score = calinski_harabasz_score(X, labels)
    avg_intracluster_distance = calculate_intracluster_distance(
        X, labels, clusterer.cluster_centers_
    )
    avg_intercluster_distance = calculate_intercluster_distance(
        X, labels, clusterer.cluster_centers_
    )
    console.log(f"Avg. Silhouette score: {avg_silhouette_score:.3f}")
    console.log(f"Avg. Calinski-Harabasz score: {avg_calinski_harabasz_score:.3f}")
    console.log(f"Avg. Intracluster distance: {avg_intracluster_distance:.3f}")
    console.log(f"Avg. Intercluster distance: {avg_intercluster_distance:.3f}")

    # append cluster label to each example
    for idx, example in enumerate(examples):
        example["cluster_label"] = int(labels[idx])

    # categorize examples by cluster label
    clustered_examples = defaultdict(list)
    for example in examples:
        clustered_examples[example["cluster_label"]].append(example)

    # save clustered examples to json files
    cluster_output_dir = "dataset/cluster_by_lexical"
    for cluster_label, cluster_examples in clustered_examples.items():
        # retrieve top-100 examples to print
        for example in cluster_examples[:100]:
            console.log(f" - {example['term']}: {example['definition']}")
        cluster_fpath = os.path.join(
            cluster_output_dir, f"cluster-{cluster_label}.jsonl"
        )
        os.makedirs(cluster_output_dir, exist_ok=True)
        with open(cluster_fpath, "w", encoding="utf-8") as f:
            for example in cluster_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        console.log(f"Cluster {cluster_label} saved to {cluster_fpath}")

    console.log(f"Lexical clustering complete. Results saved in {cluster_output_dir}")


def split_by_semantic(args: Namespace):
    model = SentenceTransformer(args.model, trust_remote_code=True, device="cuda:0")
    embedding_fpath = f"dataset/{args.dataset}.{args.split}.json"
    examples = get_random_example_word_definition(args.dataset, args.sample_num)

    console.log(examples[:10])

    if os.path.isfile(embedding_fpath):
        console.log("Loading existed file:", embedding_fpath)
        with open(embedding_fpath, "r") as f:
            examples = json.load(f)
    else:
        console.log("Computing embeddings from Model ...")
        texts = [
            example["instruction"] + "\n\n" + example["definition"]
            for example in examples
        ]
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=16,
            device="cuda",
            show_progress_bar=True,
        ).tolist()
        for idx in range(len(examples)):
            example = examples[idx]
            embedding = embeddings[idx]
            example["embedding"] = embedding

        with open(embedding_fpath, "w") as f:
            json.dump(examples, f, ensure_ascii=False, indent=4)

    embeddings = np.array([example["embedding"] for example in examples])

    # Cluster dataset to several clusters by trained k-means clusterer
    clusterer = load_clusterer(args)
    labels = clusterer.fit_predict(embeddings)  # kmeans labels

    # calculate silhouette score
    avg_silhouette_score = silhouette_score(embeddings, labels, random_state=42)
    avg_calinski_harabasz_score = calinski_harabasz_score(embeddings, labels)
    avg_intracluster_distance = calculate_intracluster_distance(
        embeddings, labels, clusterer.cluster_centers_
    )
    avg_intercluster_distance = calculate_intercluster_distance(
        embeddings, labels, clusterer.cluster_centers_
    )
    console.log(f"Avg. Silhouette score: {avg_silhouette_score:.3f}")
    console.log(f"Avg. Calinski-Harabasz score: {avg_calinski_harabasz_score:.3f}")
    console.log(f"Avg. Intracluster distance: {avg_intracluster_distance:.3f}")
    console.log(f"Avg. Intercluster distance: {avg_intercluster_distance:.3f}")

    cluster2examples = dict()
    for cluster in range(args.cluster_centroids):
        console.log(f"Cluster {cluster}")
        for idx in range(len(examples)):
            example = examples[idx]
            if clusterer.predict([example["embedding"]])[0] == cluster:
                if cluster not in cluster2examples:
                    cluster2examples[cluster] = []
                example["cluster_label"] = cluster
                cluster2examples[cluster].append(example.copy())
        # retrieve top-100 examples to print
        for example in cluster2examples[cluster][:100]:
            console.log(f" - {example['term']}: {example['definition']}")
        console.rule()

    cluster_output_dir = "dataset/cluster_by_semantic"
    os.makedirs(cluster_output_dir, exist_ok=True)
    for cluster, examples in cluster2examples.items():
        cluster_fpath = os.path.join(cluster_output_dir, f"cluster-{cluster}.jsonl")
        with open(cluster_fpath, "w") as f:
            for example in examples:
                example.pop("embedding")
                f.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--clusterer", type=str, default="kmeans")
    parser.add_argument(
        "--dataset", type=str, default="3D-EX"
    )  # "wordnet", "oxford", "wiki", "slang", "3D-EX"
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cluster_centroids", type=int, default=4)
    parser.add_argument("--sample_num", type=int, default=10000)
    # parser.add_argument("--sample_num", type=int, default=1000)
    parser.add_argument("--split_by", type=str, default="semantic")
    # parser.add_argument("--split_by", type=str, default="lexical")
    args = parser.parse_args()

    if args.split_by == "lexical":
        split_by_lexical(args)
    elif args.split_by == "semantic":
        split_by_semantic(args)
