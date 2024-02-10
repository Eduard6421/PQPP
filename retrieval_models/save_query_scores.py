import pickle
import pandas as pd
import numpy as np


def precision_at_k(ground_truth, retrieved_ids, k=10):
    """
    Computes the precision at K for multiple queries.

    :param ground_truth: Dictionary where keys are query IDs and values are arrays of relevant image IDs.
    :param retrieved_ids: Dictionary where keys are query IDs and values are arrays of retrieved image IDs.
    :param k: The number of top items to consider for precision calculation.
    :return: The average precision at K across all queries.
    """
    id_to_score = {}
    total_precision = 0
    num_queries = 0

    # Iterate over each query using the keys (query IDs)
    for query_id, relevant_ids in ground_truth.items():
        # Get the retrieved IDs for the same query
        if query_id in retrieved_ids:
            retrieved_at_k = retrieved_ids[query_id][:k]
            relevant_at_k = sum([1 for id in retrieved_at_k if id in relevant_ids])
            id_to_score[query_id] = relevant_at_k / k
            # Calculate precision for this query and add to total
            total_precision += relevant_at_k / k
            num_queries += 1
        else:
            print(f"No retrieved results for query ID: {query_id}")

    # Calculate the average precision
    if num_queries == 0:
        return 0
    average_precision = total_precision / num_queries
    return id_to_score


def rank(ground_truth, retrieved_ids):
    """
    Computes the rank for multiple queries.

    :param ground_truth: Dictionary where keys are query IDs and values are arrays of relevant image IDs.
    :param retrieved_ids: Dictionary where keys are query IDs and values are arrays of retrieved image IDs.
    :return: The mean reciprocal rank across all queries.
    """
    total_reciprocal_rank = 0
    num_queries = 0

    id_to_score = {}

    # Iterate over each query using the keys (query IDs)
    for query_id, relevant_ids in ground_truth.items():
        if query_id in retrieved_ids:
            # Find the rank of the first relevant item
            for rank, id in enumerate(retrieved_ids[query_id], start=1):
                if id in relevant_ids:
                    id_to_score[query_id] = 1 / rank
                    total_reciprocal_rank += 1 / rank
                    break
            num_queries += 1
        else:
            print(f"No retrieved results for query ID: {query_id}")

    # Calculate the mean reciprocal rank
    if num_queries == 0:
        return 0
    mean_rank = total_reciprocal_rank / num_queries
    return id_to_score


def mean_average_precision_optimized(ground_truth, retrieved_ids):
    """
    Optimized version of mean average precision computation using NumPy for vectorized operations.

    :param ground_truth: Dictionary where keys are query IDs and values are arrays of relevant image IDs.
    :param retrieved_ids: Dictionary where keys are query IDs and values are arrays of retrieved image IDs.
    :return: The mean average precision across all queries.
    """
    id_to_score = {}
    total_average_precision = 0
    num_queries = 0

    for query_id, relevant_ids in ground_truth.items():
        if query_id in retrieved_ids:
            retrieved = np.array(retrieved_ids[query_id])
            relevant = np.array(list(relevant_ids))
            num_relevant = len(list(relevant))

            if num_relevant == 0:
                continue

            # Create a boolean array where True indicates a match
            is_relevant = np.in1d(retrieved, relevant)

            # Cumulative sum of the relevant items retrieved
            cumsum_relevant = np.cumsum(is_relevant)

            # Precision at each point where a relevant item was retrieved
            precision_at_relevant = (
                cumsum_relevant[is_relevant]
                / (np.arange(len(retrieved)) + 1)[is_relevant]
            )

            average_precision = np.sum(precision_at_relevant) / num_relevant
            id_to_score[query_id] = average_precision
            total_average_precision += average_precision
            num_queries += 1
        else:
            print(f"No retrieved results for query ID: {query_id}")

    return id_to_score


ground_truth_map = "../dataset/merged_retrieval_gt.pickle"
predictions = "./blip2/blip2_retrieval_results.pickle"
best_captions_path = "../dataset/best_captions_df.pickle"

ground_truth = pickle.load(open(ground_truth_map, "rb"))
retrieval_results = pickle.load(open(predictions, "rb"))
best_captions_df = pd.read_pickle(best_captions_path)

pk_scores_map = precision_at_k(ground_truth, retrieval_results, k=10)
ap_scores_map = mean_average_precision_optimized(ground_truth, retrieval_results)
mrr_scores_map = rank(ground_truth, retrieval_results)


with open("./blip2/pk_scores_map_new.pickle", "wb") as handle:
    pickle.dump(pk_scores_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("./blip2/ap_scores_map_new.pickle", "wb") as handle:
    pickle.dump(ap_scores_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("./blip2/mrr_scores_map_new.pickle", "wb") as handle:
    pickle.dump(mrr_scores_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
