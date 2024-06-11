import pickle
from scipy.stats import pearsonr, kendalltau

blip2_results_path = "../../../retrieval_models/blip2/blip2_retrieval_results.pickle"
blip2_noised_results_path = (
    "../../../retrieval_models/blip2/blip2_retrieval_results_noised.pickle"
)

mrr_scores_path = (
    "../../../retrieval_models/retrieval_models_scores/avg_scores_rr.pickle"
)
p10_scores_path = (
    "../../../retrieval_models/retrieval_models_scores/avg_scores_p10.pickle"
)


TOP_K = 100

with open(mrr_scores_path, "rb") as f:
    mrr_scores = pickle.load(f)

with open(p10_scores_path, "rb") as f:
    p10_scores = pickle.load(f)

with open(blip2_results_path, "rb") as f:
    blip2_results = pickle.load(f)

with open(blip2_noised_results_path, "rb") as f:
    blip2_noised_results = pickle.load(f)


def compute_iou(list1, list2):
    list1 = list1[:TOP_K]
    list2 = list2[:TOP_K]
    intersection = len(set(list1).intersection(set(list2)))
    union = len(set(list1).union(set(list2)))
    return intersection / union


from tqdm import tqdm

iou_scores = []
for i in tqdm(range(8000, 10000)):
    score = compute_iou(blip2_results[i], blip2_noised_results[i])
    iou_scores.append(score)


# write iou scores to file

with open("iou_scores_blip2.pickle", "wb") as f:
    pickle.dump(iou_scores, f)


def compute_correlations(map1, map2, title):
    l2 = []
    for i in range(8000, 10000):
        l2.append(map2[i])
    print("============")

    print(len(map1))
    print(len(l2))

    (pearson, p_value_pearson) = pearsonr(map1, l2)
    (kendall, p_value_kendall) = kendalltau(map1, l2)

    print(title)
    print("Pearson Correlation {} p-value {} ".format(pearson, p_value_pearson))
    print("Kendall Correlation {} p-value {} ".format(kendall, p_value_kendall))
    print()


compute_correlations(iou_scores, mrr_scores, "MRR Correlations")
compute_correlations(iou_scores, p10_scores, "P10 Correlations")
