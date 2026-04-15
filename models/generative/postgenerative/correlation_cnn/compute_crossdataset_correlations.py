# Load the predictions
import pickle 
import pandas as pd
import numpy as np
selected_model = "glide"

trained_dataset = "mscoco"
other_dataset = "drawbench"
file_path = fr"C:\Users\User\Desktop\Research\PQPP\models\generative\postgenerative\correlation_cnn\crossmodel_{trained_dataset}_trained_{selected_model}_test_predictions.pickle"

with open(file_path, "rb") as f:
    predictions = pickle.load(f)

# Load the ground truth
gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{selected_model}\{selected_model}_test.csv"
gt_df = pd.read_csv(gt_path)

scores = gt_df["score"].values


# Compute the kendall tau correlation

import pandas as pd

from scipy.stats import kendalltau
from scipy.stats import pearsonr


predictions = np.array(predictions)
scores = np.array(scores)


indices = list(gt_df[gt_df['source'] == other_dataset].index)


# Compute Kendall tau correlation
kendall_tau_corr, kendall_tau_p_value = kendalltau(predictions, scores[indices])
print(f"Kendall tau correlation: {kendall_tau_corr}, p-value: {kendall_tau_p_value}")

# Compute Pearson correlation
pearson_corr, pearson_p_value = pearsonr(predictions, scores[indices])
print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p_value}")
