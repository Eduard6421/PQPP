# Compute the Kendall tau correlation and Pearson correlation between the predictions and the ground truth scores for 
# 1) Precision and Reciprocal Rank for Retrieval
# 2) Precision from Retrieval and SCore from Generative
# 3) Reciprocal Rank from Retrieval and SCore from Generative

import pandas as pd


# load for train / test /val

retrieval_avg_gt_path_train = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\average\average_train.csv"
retrieval_gt_df_train = pd.read_csv(retrieval_avg_gt_path_train)

retrieval_avg_gt_path_val = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\average\average_val.csv"
retrieval_gt_df_val = pd.read_csv(retrieval_avg_gt_path_val)

retrieval_avg_gt_path_test = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\average\average_test.csv"
retrieval_gt_df_test = pd.read_csv(retrieval_avg_gt_path_test)


# concat train val test

retrieval_gt_df = pd.concat([retrieval_gt_df_train, retrieval_gt_df_val, retrieval_gt_df_test])



# Load from train val test
generative_avg_gt_path_train = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_train.csv"
generative_gt_df_train = pd.read_csv(generative_avg_gt_path_train)

generative_avg_gt_path_val = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_val.csv"
generative_gt_df_val = pd.read_csv(generative_avg_gt_path_val)

generative_avg_gt_path_test = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_test.csv"
generative_gt_df_test = pd.read_csv(generative_avg_gt_path_test)

# concat them

generative_gt_df = pd.concat([generative_gt_df_train, generative_gt_df_val, generative_gt_df_test])




# Extract precision and reciprocal rank from retrieval ground truth

retrieval_precision = retrieval_gt_df["precision"].values
retrieval_reciprocal_rank = retrieval_gt_df["reciprocal_rank"].values

# Extract score from generative ground truth

generative_score = generative_gt_df["score"].values


# Compute pearson and kentall tau for scenario 1

from scipy.stats import kendalltau
from scipy.stats import pearsonr

print("Precision and Reciprocal Rank for Retrieval")


# Compute Pearson correlation
pearson_corr, pearson_p_value = pearsonr(retrieval_precision, retrieval_reciprocal_rank)
print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p_value}")

# Compute Kendall tau correlation
kendall_tau_corr, kendall_tau_p_value = kendalltau(retrieval_precision, retrieval_reciprocal_rank)
print(f"Kendall tau correlation: {kendall_tau_corr}, p-value: {kendall_tau_p_value}")


print(' ================================')

# Compute pearson and kentall tau for scenario 2

print("Precision from Retrieval and SCore from Generative")

# Compute Pearson correlation
pearson_corr, pearson_p_value = pearsonr(retrieval_precision, generative_score)
print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p_value}")

# Compute Kendall tau correlation
kendall_tau_corr, kendall_tau_p_value = kendalltau(retrieval_precision, generative_score)
print(f"Kendall tau correlation: {kendall_tau_corr}, p-value: {kendall_tau_p_value}")



print(' ================================')

# Compute pearson and kentall tau for scenario 3

print("Reciprocal Rank from Retrieval and SCore from Generative")

# Compute Pearson correlation
pearson_corr, pearson_p_value = pearsonr(retrieval_reciprocal_rank, generative_score)
print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p_value}")

# Compute Kendall tau correlation
kendall_tau_corr, kendall_tau_p_value = kendalltau(retrieval_reciprocal_rank, generative_score)
print(f"Kendall tau correlation: {kendall_tau_corr}, p-value: {kendall_tau_p_value}")
