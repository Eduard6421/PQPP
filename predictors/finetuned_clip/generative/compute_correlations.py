import pickle
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, kendalltau

with open("test_predictions.pickle", "rb") as f:
    predictions = pickle.load(f)


# We need to parse the predictions array. Groups of 4 corespond to a single query.
# The first 2 corespond to clip
# The other 2 corespond to blip2
# We just need to compute the average of groups of 4
def split_into_groups(list, group_size):
    return [list[i : i + group_size] for i in range(0, len(list), group_size)]


def compute_mean_for_each_group(list):
    return [np.mean(group) for group in list]


individual_predictions = split_into_groups(predictions, 4)
avg_predictions = compute_mean_for_each_group(individual_predictions)


gt_all_models_path = "../../../../dataset/gt_for_generative_all_models_new.csv"
gt_all_models = pd.read_csv(gt_all_models_path)


scores = gt_all_models["score"].tolist()


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


pickle.dump(avg_predictions, open("test_predictions_generative.pickle", "wb"))
compute_correlations(avg_predictions, scores, "Correlation AVG models")
