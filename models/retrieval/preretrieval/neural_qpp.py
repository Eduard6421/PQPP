import gensim.downloader as api
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm

nltk.download("stopwords")
nltk.download("punkt")

# Global variable for the model
model = api.load("word2vec-google-news-300")

def find_neighbours(ego_word, min_similarity, max_neighbours):
    global model
    while True:
        try:
            top_similar_words = model.similar_by_word(ego_word, topn=max_neighbours)
        except KeyError:
            # Word not in vocabulary
            return []
        if top_similar_words[-1][1] > min_similarity:
            max_neighbours += 50
        else:
            neighbors = [
                (neighbor, similarity)
                for neighbor, similarity in top_similar_words
                if similarity > min_similarity
            ]
            return neighbors

def create_ego_network_for_word(
    graph: nx.Graph,
    word: str,
    alpha: int,
    beta: float,
    current_level,
):
    global model
    if word not in graph:
        graph.add_node(word)
    try:
        _, top_similar_score = model.similar_by_word(word, topn=1)[0]
    except KeyError:
        # Word not in vocabulary
        return graph
    min_similarity = top_similar_score * beta

    neighbors = find_neighbours(word, min_similarity, 100)

    for neighbour_word, neighbour_weight in neighbors:
        if neighbour_word not in graph:
            graph.add_node(neighbour_word)
        graph.add_edge(word, neighbour_word, weight=neighbour_weight)

    if current_level < alpha:
        for neighbour_word, _ in neighbors:
            graph = create_ego_network_for_word(
                graph=graph,
                word=neighbour_word,
                alpha=alpha,
                beta=beta,
                current_level=current_level + 1,
            )

    return graph

def create_ego_network(alpha, beta, word):
    graph = nx.Graph()
    ego_network = create_ego_network_for_word(
        graph=graph, word=word, alpha=alpha, beta=beta, current_level=0
    )
    return ego_network

def remove_stopwords(sentence):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [
        w.lower() for w in word_tokens if w.isalpha() and w.lower() not in stop_words
    ]
    return filtered_sentence

def create_ego_and_compute_metrics(alpha, beta, word):
    global model
    ego_network = create_ego_network(alpha, beta, word)

    if word not in ego_network:
        # Word not processed due to absence in the model
        return {
            'edge_count': 0,
            'edge_weight_sum': 0,
            'inverse_edge_frequency': 0,
            'degree_centrality': 0,
            'closeness_centrality': 0,
            'betweenness_centrality': 0,
            'page_rank': 0,
        }

    edge_count = ego_network.number_of_edges()
    edge_weight_sum = ego_network.size(weight="weight")

    degree_centrality = ego_network.degree(word)
    inverse_edge_frequency = (
        np.log(edge_count / degree_centrality) if degree_centrality > 0 else 0
    )

    closeness_centrality = nx.closeness_centrality(ego_network, u=word)
    betweenness_centrality = nx.betweenness_centrality(ego_network, normalized=True).get(word, 0)
    page_rank = nx.pagerank(ego_network, weight="weight").get(word, 0)

    return {
        'edge_count': edge_count,
        'edge_weight_sum': edge_weight_sum,
        'inverse_edge_frequency': inverse_edge_frequency,
        'degree_centrality': degree_centrality,
        'closeness_centrality': closeness_centrality,
        'betweenness_centrality': betweenness_centrality,
        'page_rank': page_rank,
    }

def neural_qpp(sentence, alpha, beta):
    filtered_sentence = remove_stopwords(sentence)
    results = []

    for word in filtered_sentence:
        metrics = create_ego_and_compute_metrics(alpha, beta, word)
        results.append((word, metrics))

    # Compute average metrics over all words
    if results:
        average_metrics = {
            metric: sum(d[1][metric] for d in results) / len(results) for metric in results[0][1]
        }
    else:
        raise Exception("No words processed")

    return average_metrics, results


alphas = [1]
betas = [ 0.9, 0.95]

if __name__ == "__main__":
    # Load your ground truth data
    test_retrieval_gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\average\average_val.csv"
    ground_truth_test = pd.read_csv(test_retrieval_gt_path)

    for alpha in alphas:
        for beta in betas:
            results = []
            print(fr"Processing with alpha={alpha} and beta={beta}")
            for _, row in tqdm(ground_truth_test.iterrows(), total=ground_truth_test.shape[0]):
                prompt = row["prompt"]
                average_metrics, word_results = neural_qpp(
                    sentence=prompt, alpha=alpha, beta=beta
                )
                results.append(average_metrics)
            with open(fr"./results_{alpha}_{beta}_val", "wb") as f:
                pickle.dump(results, f)
