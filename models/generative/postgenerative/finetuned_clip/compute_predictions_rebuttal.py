# Load the predictions
import pickle 
import pandas as pd
import numpy as np
selected_model = "glide"

trained_dataset = "mscoco"
other_dataset = "mscoco"
file_path = fr"C:\Users\User\Desktop\Research\PQPP\models\generative\postgenerative\finetuned_clip\rebuttal_model_{trained_dataset}_{selected_model}_test_predictions.pickle"

with open(file_path, "rb") as f:
    predictions = pickle.load(f)

def split_into_groups(list, group_size):
    return [list[i : i + group_size] for i in range(0, len(list), group_size)]

def compute_mean_for_each_group(list):
    return [np.mean(group) for group in list]

predictions = split_into_groups(predictions, 4)
predictions = compute_mean_for_each_group(predictions)


# Load the ground truth
gt_path = fr"C:\Users\User\Desktop\Research\PQPP\rebuttal\automatic_test_gt.csv"
gt_df = pd.read_csv(gt_path)

scores = gt_df["sdxl_new"].values


# Compute the kendall tau correlation

import pandas as pd

from scipy.stats import kendalltau
from scipy.stats import pearsonr


predictions = np.array(predictions)
scores = np.array(scores)


# Compute Kendall tau correlation
kendall_tau_corr, kendall_tau_p_value = kendalltau(predictions, scores)
print(f"Kendall tau correlation: {kendall_tau_corr}, p-value: {kendall_tau_p_value}")

# Compute Pearson correlation
pearson_corr, pearson_p_value = pearsonr(predictions, scores)
print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p_value}")
