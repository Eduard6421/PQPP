import pandas as pd
import gensim.downloader as api
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import multiprocessing
import numpy as np
import concurrent.futures
import networkx as nx
import numpy as np
from nltk.corpus import stopwords
import pickle
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import string
import re
import nltk
import concurrent.futures
import threading

progress_lock = threading.Lock()
progress_counter = 0

# dataframe that contains the best caption for each item
best_caption_path = "../../../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path).head(10000)

train_df = best_captions_df.iloc[0:6000]
validation_df = best_captions_df[6000:8000]
test_df = best_captions_df[8000:10000]
gt_all_models_path = "../../../dataset/gt_for_generative_all_models.csv"
gt_all_models = pd.read_csv(gt_all_models_path)
model = api.load("word2vec-google-news-300")

stopwords = set(stopwords.words("english"))


def find_neighbours(ego_word, min_similarity, max_neighbours, model):
    top_similar_words = model.similar_by_word(ego_word, topn=max_neighbours)
    if top_similar_words[-1][1] > min_similarity:
        print("requested for more")
        return find_neighbours(ego_word, min_similarity, max_neighbours + 5000, model)
    else:
        neighbors = [
            (neighbor, similarity)
            for neighbor, similarity in top_similar_words
            if similarity > min_similarity
        ]
        return neighbors


def create_ego_network_for_word(
    graph: nx.Graph,
    parsed_set: set(),
    word: str,
    alpha: int,
    beta: float,
    current_level,
    model,
):
    if word in parsed_set:
        return (False, False)

    if not word in model:
        return (False, False)

    if not word in graph:
        graph.add_node(word)
        parsed_set.add(word)

    _, top_similar_score = model.similar_by_word(word, topn=1)[0]
    min_similarity = top_similar_score * beta

    neighbors = find_neighbours(word, min_similarity, 5000, model=model)

    for neighbour_word, neighbour_weight in neighbors:
        if not neighbour_word in graph:
            graph.add_node(neighbour_word)
        graph.add_edge(word, neighbour_word, weight=neighbour_weight)

    if current_level < alpha:
        for neighbour_word, neighbour_weight in neighbors:
            graph, parsed_set = create_ego_network_for_word(
                graph=graph,
                word=neighbour_word,
                alpha=alpha,
                beta=beta,
                current_level=current_level + 1,
                model=model,
                parsed_set=parsed_set,
            )

    return graph, parsed_set


def create_ego_network(alpha, beta, word, model):
    graph = nx.Graph()
    current_words = set([])
    ego_network, _parsed_set = create_ego_network_for_word(
        graph=graph,
        word=word,
        alpha=alpha,
        beta=beta,
        current_level=0,
        model=model,
        parsed_set=current_words,
    )
    return ego_network


def remove_stopwords(sentence):
    word_tokens = word_tokenize(sentence)
    # Regular expression pattern for matching tokens made only of punctuation
    punctuation_pattern = f"^[{re.escape(string.punctuation)}]+$"

    # Filter out stopwords
    filtered_sentence = [w for w in word_tokens if w.lower() not in stopwords]

    # Filter out tokens that are only punctuation
    tokens_without_punctuation = [
        token for token in filtered_sentence if not re.match(punctuation_pattern, token)
    ]
    return tokens_without_punctuation


def create_ego_and_compute_metrics(alpha, beta, word, model):
    ego_network = create_ego_network(alpha, beta, word, model)

    if ego_network == False:
        return (False, False)

    edge_count = ego_network.number_of_edges()
    edge_weight_sum = ego_network.size(weight="weight")

    degree_centrality = len(ego_network.edges(word))
    inverse_edge_frequency = (
        np.log(edge_count / degree_centrality) if degree_centrality > 0 else 0
    )

    closeness_centrality = nx.closeness_centrality(ego_network, word)
    between_centrality = nx.betweenness_centrality(ego_network)[word]
    page_rank = nx.pagerank(ego_network, weight="weight")[word]

    return (
        True,
        {
            "edge_count": edge_count,
            "edge_weight_sum": edge_weight_sum,
            "inverse_edge_frequency": inverse_edge_frequency,
            "degree_centrality": degree_centrality,
            "closeness_centrality": closeness_centrality,
            "between_centrality": between_centrality,
            "page_rank": page_rank,
        },
    )


def neural_qpp(sentence, alpha, beta):
    global model
    # Remove stopwrods from sentence
    filtered_sentence = remove_stopwords(sentence)
    args = [(alpha, beta, word, model) for word in filtered_sentence]

    # with multiprocessing.Pool(processes=16) as pool:
    #    results = pool.starmap(create_ego_and_compute_metrics, args)
    results = list(map(lambda arg: create_ego_and_compute_metrics(*arg), args))

    collected_items = [item[1] for item in results if item[0] is True]

    average_metrics = {
        metric: sum(d[metric] for d in collected_items) / len(collected_items)
        for metric in collected_items[0]
    }

    return average_metrics


def process_row(args):
    global progress_counter
    idx, row, alpha, beta, gt_df = args  # Unpack arguments
    image_id = row["image_id"]
    caption = row["best_caption"]
    gt_match = gt_df[gt_df["image_id"] == image_id]
    if len(gt_match) > 1:
        raise Exception("Multiple ground truths found for a single image_id")
    neural_qpp_metrics = neural_qpp(caption, alpha, beta)
    with progress_lock:
        progress_counter += 1
        print(f"Progress: {progress_counter} rows processed", end="\r")
    return neural_qpp_metrics


def search_hyperparams(gt_df, validation_df, split):
    all_results = []
    global progress_counter
    # alpha: 2, beta: 0.9
    for alpha in [1, 2]:
        for beta in [0.9375]:
            progress_counter = 0

            # Prepare data for parallel processing
            args = [
                (enum, row, alpha - 1, beta, gt_df)
                for enum, (idx, row) in enumerate(validation_df.iterrows())
            ]
            results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                # The map function applies 'process_row' to each item in 'args', maintaining the order of results
                chunk_results = list(
                    executor.map(process_row, args)
                )  # Convert the map object to a list to evaluate all results

                results.extend(
                    chunk_results
                )  # Extend the main results list with the results from this chunk

            print(f"Completed jobs for alpha={alpha}, beta={beta}: {len(results)}")

            # Save the results for the current alpha and beta
            with open(
                f"gt_all_models_alpha_{alpha}_beta_{beta}_{split}.pkl", "wb"
            ) as file:
                pickle.dump(results, file)  # Dump 'results'

            all_results.extend(
                results
            )  # Optionally, collect all results across all alphas and betas

    # Return the collected results after all iterations
    return all_results


if __name__ == "__main__":
    search_hyperparams(gt_all_models, test_df, "test")
    search_hyperparams(gt_all_models, validation_df, "validation")
