import pickle
import pandas as pd
from scipy.stats import pearsonr, kendalltau


with open("test_predictions_mrr.pickle", "rb") as f:
    predictions = pickle.load(f)
"""

"""

with open("../../../dataset/retrieval_models_scores/avg_scores_rr.pickle", "rb") as f:
    scores = pickle.load(f)


def compute_correlations(map1, map2, title):
    l1 = []
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


compute_correlations(predictions, scores, "MCorrelation AVG models ")
