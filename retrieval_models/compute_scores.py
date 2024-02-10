import pickle
import pandas as pd


def average_precision_at_k(ground_truth, retrieved_ids, k=10):
    """
    Computes the average precision at K for multiple queries.

    :param ground_truth: Dictionary where keys are query IDs and values are arrays of relevant image IDs.
    :param retrieved_ids: Dictionary where keys are query IDs and values are arrays of retrieved image IDs.
    :param k: The number of top items to consider for precision calculation.
    :return: The average precision at K across all queries.
    """
    total_precision = 0
    num_queries = 0

    # Iterate over each query using the keys (query IDs)
    for query_id, relevant_ids in ground_truth.items():
        # Get the retrieved IDs for the same query
        if query_id in retrieved_ids:
            retrieved_at_k = retrieved_ids[query_id][:k]
            relevant_at_k = sum([1 for id in retrieved_at_k if id in relevant_ids])

            # Calculate precision for this query and add to total
            total_precision += relevant_at_k / k
            num_queries += 1
        else:
            print(f"No retrieved results for query ID: {query_id}")

    # Calculate the average precision
    if num_queries == 0:
        return 0
    average_precision = total_precision / num_queries
    return average_precision


def mean_reciprocal_rank(ground_truth, retrieved_ids):
    """
    Computes the mean reciprocal rank for multiple queries.

    :param ground_truth: Dictionary where keys are query IDs and values are arrays of relevant image IDs.
    :param retrieved_ids: Dictionary where keys are query IDs and values are arrays of retrieved image IDs.
    :return: The mean reciprocal rank across all queries.
    """
    total_reciprocal_rank = 0
    num_queries = 0

    # Iterate over each query using the keys (query IDs)
    for query_id, relevant_ids in ground_truth.items():
        if query_id in retrieved_ids:
            # Find the rank of the first relevant item
            for rank, id in enumerate(retrieved_ids[query_id], start=1):
                if id in relevant_ids:
                    total_reciprocal_rank += 1 / rank
                    break
            num_queries += 1
        else:
            print(f"No retrieved results for query ID: {query_id}")

    # Calculate the mean reciprocal rank
    if num_queries == 0:
        return 0
    mean_rank = total_reciprocal_rank / num_queries
    return mean_rank


def mean_average_precision_optimized(ground_truth, retrieved_ids):
    """
    Optimized version of mean average precision computation using NumPy for vectorized operations.

    :param ground_truth: Dictionary where keys are query IDs and values are arrays of relevant image IDs.
    :param retrieved_ids: Dictionary where keys are query IDs and values are arrays of retrieved image IDs.
    :return: The mean average precision across all queries.
    """
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
            total_average_precision += average_precision
            num_queries += 1
        else:
            print(f"No retrieved results for query ID: {query_id}")

    return total_average_precision / num_queries if num_queries > 0 else 0


ground_truth_map = "../dataset/merged_retrieval_gt.pickle"
predictions = "./clip/clip_retrieval_results.pickle"
best_captions_path = "../dataset/best_captions_df.pickle"

ground_truth = pickle.load(open(ground_truth_map, "rb"))
retrieval_results = pickle.load(open(predictions, "rb"))
best_captions_df = pd.read_pickle(best_captions_path)
import numpy as np


print(
    "Precision at 10: ", average_precision_at_k(ground_truth, retrieval_results, k=10)
)
# print(
#    "Precision at 100: ", average_precision_at_k(ground_truth, retrieval_results, k=100)
# )
# print("MAP", mean_average_precision_optimized(ground_truth, retrieval_results))
print("MRR", mean_reciprocal_rank(ground_truth, retrieval_results))
