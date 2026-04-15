# Load the predictions
import pickle 
import pandas as pd
import numpy as np

selected_model = "clip"
trained_dataset = "mscoco"

other_dataset = "mscoco"
selected_metric = "mrr"

metric = {
    "mrr": fr"{selected_model}_rr",
    "p10": fr"{selected_model}_pk"
}

#caption_id,caption,sdxl_new,glide_new,blip2_rr,blip2_pk,clip_rr,clip_pk

file_path = fr"./rebuttal_{trained_dataset}_{selected_model}_{selected_metric}_predictions.pickle"
# C:\Users\User\Desktop\Research\PQPP\models\retrieval\postretrieval\correlation_cnn\crossmodel_drawbench_blip2_precision_best_model.pth
with open(file_path, "rb") as f:
    predictions = pickle.load(f)

#def split_into_groups(list, group_size):
#    return [list[i : i + group_size] for i in range(0, len(list), group_size)]

#def compute_mean_for_each_group(list):
#    return [np.mean(group) for group in list]

#predictions = split_into_groups(predictions, 4)
#predictions = compute_mean_for_each_group(predictions)



# Load the ground truth
gt_path = fr"C:\Users\User\Desktop\Research\PQPP\rebuttal\automatic_test_gt.csv"
gt_df = pd.read_csv(gt_path)

scores = gt_df[metric[selected_metric]].values


# Compute the kendall tau correlation

import pandas as pd

from scipy.stats import kendalltau
from scipy.stats import pearsonr


predictions = np.array(predictions)
scores = np.array(scores)


#indices = list(gt_df[gt_df['source'] == other_dataset].index)


print("Model", selected_model, "Dataset", trained_dataset, "Metric", selected_metric)

# Compute Pearson correlation
pearson_corr, pearson_p_value = pearsonr(predictions, scores)
print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p_value}")


# Compute Kendall tau correlation
kendall_tau_corr, kendall_tau_p_value = kendalltau(predictions, scores)
print(f"Kendall tau correlation: {kendall_tau_corr}, p-value: {kendall_tau_p_value}")
