import pickle
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, kendalltau

with open("test_predictions.pickle", "rb") as f:
    predictions = pickle.load(f)

with open("../../../retrieval_models/avg_scores_mrr_new.pickle", "rb") as f:
    mrr_scores = pickle.load(f)

with open("../../../retrieval_models/avg_scores_p10_new.pickle", "rb") as f:
    p10_scores = pickle.load(f)


def split_into_groups(list, group_size):
    return [list[i : i + group_size] for i in range(0, len(list), group_size)]


def compute_metric_of_each_query(group):
    # cut index should be half on group length
    cut_index = len(group) // 2
    clip_results = group[:cut_index]
    blip_results = group[cut_index:]

    # Compute p@10 for each one.
    # Count how many values higher than 0.5 are in the first 10 clip_results

    clip_p10 = 0
    blip_p10 = 0

    for i in range(10):
        print(clip_results[i])
        print(blip_results[i])
        if clip_results[i] >= 0.5:
            clip_p10 += 1
        if blip_results[i] >= 0.5:
            blip_p10 += 1

    print(f"blip_score_per_division : {blip_p10} , clip_scoer_pre_division: {clip_p10}")

    blip_p10 /= 10
    clip_p10 /= 10

    print(
        f"blip_score_after_division : {blip_p10} , clip_scoer_after_division: {clip_p10}"
    )

    # Compute the MRR for each one

    clip_mrr = 0
    blip_mrr = 0

    for i in range(len(clip_results)):
        if clip_results[i] >= 0.5:
            clip_mrr = 1 / (i + 1)
            break

    for i in range(len(blip_results)):
        if blip_results[i] >= 0.5:
            blip_mrr = 1 / (i + 1)
            break

    avg_p10 = np.mean([clip_p10, blip_p10])
    avg_mrr = np.mean([clip_mrr, blip_mrr])

    return (avg_p10, avg_mrr)


def collect_metrics(all_metrics):
    p10s = [metric[0] for metric in all_metrics]
    mrrs = [metric[1] for metric in all_metrics]
    return p10s, mrrs


def compute_correlations(map1, map2, title):
    l2 = []
    for i in range(8000, 10000):
        l2.append(map2[i])

    print(len(map1))
    print(len(l2))

    (pearson, p_value_pearson) = pearsonr(map1, l2)
    (kendall, p_value_kendall) = kendalltau(map1, l2)

    print(title)
    print("Pearson Correlation {} p-value {} ".format(pearson, p_value_pearson))
    print("Kendall Correlation {} p-value {} ".format(kendall, p_value_kendall))
    print()


query_results = split_into_groups(predictions, 50)
metrics = [compute_metric_of_each_query(group) for group in query_results]
predicted_p10, predicted_mrr = collect_metrics(metrics)

with open("predicted_p10.pickle", "wb") as f:
    pickle.dump(predicted_p10, f)

compute_correlations(predicted_p10, p10_scores, "P10 Correlations")
compute_correlations(predicted_mrr, mrr_scores, "MRR Correlations")
