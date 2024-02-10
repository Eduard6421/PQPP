import gensim.downloader as api
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import multiprocessing
import numpy as np

# nltk.download("stopwords")


def find_neighbours(ego_word, min_similarity, max_neighbours, model):
    top_similar_words = model.similar_by_word(ego_word, topn=max_neighbours)
    if top_similar_words[-1][1] > min_similarity:
        return find_neighbours(ego_word, min_similarity, max_neighbours + 50, model)
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
    model,
):
    if not word in graph:
        graph.add_node(word)

    _, top_similar_score = model.similar_by_word(word, topn=1)[0]
    min_similarity = top_similar_score * beta

    neighbors = find_neighbours(word, min_similarity, 100, model=model)
    print(neighbors)

    for neighbour_word, neighbour_weight in neighbors:
        if not neighbour_word in graph:
            graph.add_node(neighbour_word)
        graph.add_edge(word, neighbour_word, weight=neighbour_weight)

    if current_level < alpha:
        for neighbour_word, neighbour_weight in neighbors:
            graph = create_ego_network_for_word(
                graph=graph,
                word=neighbour_word,
                alpha=alpha,
                beta=beta,
                current_level=current_level + 1,
                model=model,
            )

    return graph


def create_ego_network(alpha, beta, word, model):
    graph = nx.Graph()
    ego_network = create_ego_network_for_word(
        graph=graph, word=word, alpha=alpha, beta=beta, current_level=0, model=model
    )
    return ego_network


def remove_stopwords(sentence):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence


def create_ego_and_compute_metrics(alpha, beta, word, model):
    ego_network = create_ego_network(alpha, beta, word, model)

    edge_count = ego_network.number_of_edges()
    edge_weight_sum = ego_network.size(weight="weight")

    degree_centrality = ego_network.number_of_edges(word)
    inverse_edge_frequency = (
        np.log(edge_count / degree_centrality) if degree_centrality > 0 else 0
    )

    closeness_centrality = nx.closeness_centrality(ego_network, word)
    between_centrality = nx.betweenness_centrality(ego_network)[word]
    page_rank = nx.pagerank(ego_network, weight="weight")[word]

    return {
        edge_count: edge_count,
        edge_weight_sum: edge_weight_sum,
        inverse_edge_frequency: inverse_edge_frequency,
        degree_centrality: degree_centrality,
        closeness_centrality: closeness_centrality,
        between_centrality: between_centrality,
        page_rank: page_rank,
    }


def neural_qpp(sentence, alpha, beta, model):
    # Remove stopwrods from sentence
    filtered_sentence = remove_stopwords(sentence)

    print(filtered_sentence)

    args = [(alpha, beta, word, model) for word in filtered_sentence]

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.starmap(create_ego_and_compute_metrics, args)

    for metric in results[0]:
        print(metric)
        print(results)

    average_metrics = {
        metric: sum(d[metric] for d in results) / len(results) for metric in results[0]
    }

    return average_metrics


if __name__ == "__main__":
    model = api.load("word2vec-google-news-300")
    results = neural_qpp(sentence="coat", alpha=1, beta=0.9, model=model)

"""
# Example usage: Create an ego network for a sample term


# Calculating centrality metrics



centrality_metrics
"""
