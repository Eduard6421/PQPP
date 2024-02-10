import pickle
import pandas as pd
from scipy.stats import pearsonr, kendalltau


with open("test_predictions.pickle", "rb") as f:
    predictions = pickle.load(f)


gt_all_models_path = "../../../../dataset/gt_for_generative_all_models_new.csv"
gt_all_models = pd.read_csv(gt_all_models_path)


scores = gt_all_models["score"].tolist()


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
