# Load the predictions
import pickle 
import pandas as pd

selected_model = "blip2"
other_model = "blip2"
task = "mrr"

metric = {
    "mrr": "reciprocal_rank",
    "p10" : "precision"
}

file_bindings = metric[task]

file_path = fr"C:\Users\User\Desktop\Research\PQPP\models\retrieval\postretrieval\finetuned_clip\{selected_model}_{task}_predictions.pickle"
with open(file_path, "rb") as f:
    predictions = pickle.load(f)

# Load the ground truth
gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\{other_model}\{other_model}_retrieval_test_results.csv"
gt_df = pd.read_csv(gt_path)
scores = gt_df[metric[task]].values


# Compute the kendall tau correlation

import pandas as pd

from scipy.stats import kendalltau
from scipy.stats import pearsonr
import numpy as np
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


# Compute Kendall tau correlation for mscoco
print("MSCOCO")	
kendall_tau_corr, kendall_tau_p_value = kendalltau(predictions[mscoco_indices], scores[mscoco_indices])
print(f"Kendall tau correlation for mscoco: {kendall_tau_corr}, p-value: {kendall_tau_p_value}")

# Compute Pearson correlation for mscoco
pearson_corr, pearson_p_value = pearsonr(predictions[mscoco_indices], scores[mscoco_indices])
print(f"Pearson correlation for mscoco: {pearson_corr}, p-value: {pearson_p_value}")

print("DRAWBENCH")	
# Compute Kendall tau correlation for drawbench
kendall_tau_corr, kendall_tau_p_value = kendalltau(predictions[drawbench_indices], scores[drawbench_indices])
print(f"Kendall tau correlation for drawbench: {kendall_tau_corr}, p-value: {kendall_tau_p_value}")

# Compute Pearson correlation for drawbench
pearson_corr, pearson_p_value = pearsonr(predictions[drawbench_indices], scores[drawbench_indices])
print(f"Pearson correlation for drawbench: {pearson_corr}, p-value: {pearson_p_value}")

print(predictions[drawbench_indices])