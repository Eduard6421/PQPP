# open test_predictions from local file test_predictions.pickle
import pickle
import pandas as pd
import numpy as np
selected_model = "sdxl"
other_model = "blip2"	

with open(fr"{selected_model}_test_predictions.pickle", "rb") as f:	
    test_predictions = pickle.load(f)


# Load the ground truth
gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\{other_model}\{other_model}_retrieval_test_results.csv"

gt_df = pd.read_csv(gt_path)

scores = gt_df["reciprocal_rank"].values



def split_into_groups(list, group_size):
    return [list[i : i + group_size] for i in range(0, len(list), group_size)]

def compute_mean_for_each_group(list):
    return [np.mean(group) for group in list]

predictions = split_into_groups(test_predictions, 4)
predictions = compute_mean_for_each_group(predictions)

# print kendall tau and personr correlations and their p values

from scipy.stats import kendalltau
from scipy.stats import pearsonr

sources = gt_df['source']

# source mscoco

mscoco_indices = list(sources[sources == 'mscoco'].index)

# source drawbench
drawbench_indices =list( sources[sources == 'drawbench'].index)

# convert predictions and scores to numpy arrays

predictions = np.array(predictions)
scores = np.array(scores)



print("MSCOCO + DRAWBENCH")
# Compute Kendall tau correlation
kendall_tau_corr, kendall_tau_p_value = kendalltau(predictions, scores)
print(f"Kendall tau correlation: {kendall_tau_corr}, p-value: {kendall_tau_p_value}")

# Compute Pearson correlation
pearson_corr, pearson_p_value = pearsonr(predictions, scores)
print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p_value}")