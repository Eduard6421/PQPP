# open test_predictions from local file test_predictions.pickle
import pickle
import pandas as pd
import numpy as np
selected_model = "sdxl"
other_model = "sdxl"	

with open(fr"{selected_model}_test_predictions.pickle", "rb") as f:	
    test_predictions = pickle.load(f)


gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{other_model}\{other_model}_test.csv"
gt_df = pd.read_csv(gt_path)

scores = gt_df["score"].values


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

'''
def generate_prediction(prediction_list):

    clip_p10_predictions = []
    clip_mrr_predictions = []
        
    blip2_p10_predictions = []
    blip2_mrr_predictions = []


    for i in range(0, len(prediction_list), 50):


        clip_current_mrr = 0

        for j in range(i, i+25):
            if(prediction_list[j] >= 0.5):
                clip_current_mrr = 1 / (j-i+1)
                break
                
        clip_match_count = 0
        for j in range(i, i+10):
            if(prediction_list[j] >= 0.5):
                clip_match_count +=1
        clip_match_count /= 10


        blip2_current_mrr = 0
        blip2_match_count = 0


        for j in range(i+25, i+50):
            if(prediction_list[j] >= 0.5):
                blip2_current_mrr = 1 / (j-i+1)
                break

        for j in range(i+25, i+35):
            if(prediction_list[j] >= 0.5):
                blip2_match_count +=1

        blip2_match_count /= 10

        clip_p10_predictions.append(clip_match_count)
        clip_mrr_predictions.append(clip_current_mrr)

        blip2_p10_predictions.append(blip2_match_count)
        blip2_mrr_predictions.append(blip2_current_mrr)

    return  clip_p10_predictions, clip_mrr_predictions, blip2_p10_predictions, blip2_mrr_predictions


clip_p10_predictions, clip_mrr_predictions, blip2_p10_predictions, blip2_mrr_predictions = generate_prediction(test_predictions)

# Save the predictions to a pickle file

with open("clip_p10_predictions.pickle", "wb") as file:
    pickle.dump(clip_p10_predictions, file)

with open("clip_mrr_predictions.pickle", "wb") as file:
    pickle.dump(clip_mrr_predictions, file)	

with open("blip2_p10_predictions.pickle", "wb") as file:
    pickle.dump(blip2_p10_predictions, file)	

with open("blip2_mrr_predictions.pickle", "wb") as file:
    pickle.dump(blip2_mrr_predictions, file)
'''