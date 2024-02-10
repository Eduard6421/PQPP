import pickle
import numpy as np
from scipy.stats import pearsonr, kendalltau

clip_scores = pickle.load(open("clip_retrieval_results.pickle", "rb"))
blip2_scores = pickle.load(open("blip2_retrieval_results.pickle", "rb"))


def average_score(clip_scores, blip2_scores):
    avg_score = {}
    for key in clip_scores:
        clip_score = clip_scores[key]
        blip2_score = blip2_scores[key]
        avg_score[key] = (clip_score + blip2_score) / 2
    return avg_score


def get_top_k_variance(obj, top_k):
    for key in obj:
        array = obj[key]
        array = array[:top_k]
        var = np.var(array)
        obj[key] = var
    return obj


avg_scores = average_score(clip_scores, blip2_scores)
blip2_top_100 = get_top_k_variance(blip2_scores, 100)
avg_top_100 = get_top_k_variance(avg_scores, 100)

# Compute variance for each item in blip2_top_100. This is a list of lists
blip2_vars = [np.var(item) for item in blip2_top_100]
avg_vars = [np.var(item) for item in avg_top_100]

avg_scores_mrr = pickle.load(
    open("../../../retrieval_models/avg_scores_mrr_new.pickle", "rb")
)
avg_scores_p10 = pickle.load(
    open("../../../retrieval_models/avg_scores_p10_new.pickle", "rb")
)

# mrr_scores_map = pickle.load(open("mrr_scores_map.pickle", "rb"))
# pk_scores_map = pickle.load(open("pk_scores_map.pickle", "rb"))


def compute_correlations(map1, map2, title):
    l1 = []
    l2 = []
    for i in range(8000, 10000):
        l1.append(map1[i])
        l2.append(map2[i])

    (pearson, p_value_pearson) = pearsonr(l1, l2)
    (kendall, p_value_kendall) = kendalltau(l1, l2)

    print(title)
    print("Pearson Correlation {} p-value {} ".format(pearson, p_value_pearson))
    print("Kendall Correlation {} p-value {} ".format(kendall, p_value_kendall))
    print()


compute_correlations(
    avg_scores_p10, avg_top_100, "Mean Variance on Avg model with AVG P@10"
)
compute_correlations(
    avg_scores_mrr, avg_top_100, "Mean Variance on Avg model with AVG MRR"
)


# compute_correlations(
#    mrr_scores_map, blip2_top_100, "Mean Variance on Best model with MRR"
# )
# compute_correlations(
#    pk_scores_map, blip2_top_100, "Mean Variance on Best model with P@10"
# )
