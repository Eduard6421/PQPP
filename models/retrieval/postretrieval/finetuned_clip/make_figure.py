# Load the predictions
import pickle 
import pandas as pd

selected_model = "clip"
other_model = "clip"
task = "p10"

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


# do a boxplot of the predictions and scores and save it in a pdf


import plotly.express as px
import pandas as pd
import plotly.io as pio

# Create a DataFrame for the boxplot
data = pd.DataFrame({
    'Predictions': predictions,
    'Scores': scores
})


# Create a Plotly boxplot treating Predictions as a continuous variable
fig = px.box(
    data, 
    x='Scores', 
    y='Predictions',
    title='Boxplot of CLIP Finetuned predictions and the P@10 scores for the CLIP retrieval model',
    labels={'Predictions': 'Predicted Score', 'Scores': 'Ground Truth Scores'},
    width=600, 
    height=400
)
# Update layout for better visualization and to remove grid lines
fig.update_layout(
    font=dict(
        size=16  # Increase font size
    ),
    title_font=dict(
        size=16  # Larger title font size
    ),
    xaxis_title_font=dict(
        size=16  # Larger x-axis title font
    ),
    yaxis_title_font=dict(
        size=16  # Larger y-axis title font
    ),
    xaxis=dict(
        showgrid=False,  # Remove grid lines from x-axis
        zeroline=False   # Remove zero line from x-axis
    ),
    yaxis=dict(
        showgrid=False,  # Remove grid lines from y-axis
        zeroline=False   # Remove zero line from y-axis
    )
)




import numpy as np



# Show the plot in a window
pio.show(fig)

# Save as a PDF file
fig.write_image('./clip_p10_clip_finetuned.pdf', format='pdf')
