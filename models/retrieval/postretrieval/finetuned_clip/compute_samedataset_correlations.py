# Load the predictions
import pickle 
import pandas as pd
import numpy as np

selected_model = "blip2"
trained_dataset = "mscoco"

other_dataset = "mscoco"
selected_metric = "mrr"

metric = {
    "mrr": "reciprocal_rank",
    "p10": "precision"
}


file_path = fr"./same_dataset_{trained_dataset}_{selected_model}_{selected_metric}_predictions.pickle"
# C:\Users\User\Desktop\Research\PQPP\models\retrieval\postretrieval\correlation_cnn\crossmodel_drawbench_blip2_precision_best_model.pth
with open(file_path, "rb") as f:
    predictions = pickle.load(f)


#def split_into_groups(list, group_size):
#    return [list[i : i + group_size] for i in range(0, len(list), group_size)]

#def compute_mean_for_each_group(list):
#    return [np.mean(group) for group in list]

#predictions = split_into_groups(predictions, 4)
#predictions = compute_mean_for_each_group(predictions)

print(len(predictions))


# Load the ground truth
gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\{selected_model}\{selected_model}_retrieval_test_results.csv"
gt_df = pd.read_csv(gt_path)

scores = gt_df[metric[selected_metric]].values


# Compute the kendall tau correlation

import pandas as pd

from scipy.stats import kendalltau
from scipy.stats import pearsonr


predictions = np.array(predictions)
scores = np.array(scores)


indices = list(gt_df[gt_df['source'] == other_dataset].index)


# Compute Pearson correlation
pearson_corr, pearson_p_value = pearsonr(predictions, scores[indices])
print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p_value}")


# Compute Kendall tau correlation
kendall_tau_corr, kendall_tau_p_value = kendalltau(predictions, scores[indices])
print(f"Kendall tau correlation: {kendall_tau_corr}, p-value: {kendall_tau_p_value}")
