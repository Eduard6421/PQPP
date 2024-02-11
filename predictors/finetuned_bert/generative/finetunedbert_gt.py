# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: PyTorch 1.13 (Local)
#     language: python
#     name: pytorch-1-13
# ---

# +
# # %pip install -U kaleido

# +
# # %pip install transformers
# -

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
import pandas as pd
import numpy as np
import pickle
from transformers import BertModel, BertTokenizer
from torch.nn import Module, Linear, Dropout
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch
from transformers import BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import scipy.stats


gt_for_generative_all_models_df =pd.read_csv('gt_for_generative_all_models_new.csv')

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go

gt_for_generative_all_models_df['score'].min()
gt_for_generative_all_models_df['normalized_score'] = (gt_for_generative_all_models_df['score'] + 1)/3

total_size = len(gt_for_generative_all_models_df)
train_size = int(total_size * 0.6)
eval_size = int(total_size * 0.2)
train_data = gt_for_generative_all_models_df.iloc[:train_size]
eval_data = gt_for_generative_all_models_df.iloc[train_size:train_size + eval_size]
test_data = gt_for_generative_all_models_df.iloc[train_size + eval_size:]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def tokenize_and_prepare(dataframe):
    inputs = tokenizer(list(dataframe['best_caption']), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(dataframe['normalized_score'].values).unsqueeze(-1).float()
    return inputs['input_ids'], inputs['attention_mask'], labels


train_inputs, train_masks, train_labels = tokenize_and_prepare(train_data)
eval_inputs, eval_masks, eval_labels = tokenize_and_prepare(eval_data)
test_inputs, test_masks, test_labels = tokenize_and_prepare(test_data)

test_scores_list = test_data['score'].tolist()


# +
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
eval_dataset = TensorDataset(eval_inputs, eval_masks, eval_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)


# -

class BertRegressor(nn.Module):
    def __init__(self, n_outputs=1):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.linear2 = nn.Linear(512, n_outputs)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.linear1(pooled_output)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.relu(x)
        return x
    
    def get_embeddings(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.linear1(pooled_output)
        return x



param_grid = {
    'learning_rate': [1e-5, 1e-4, 5e-5],
    'num_epochs': [15],
    'weight_decay': [0, 0.1, 0.01]
}


def evaluate_model(model, data_loader):
    model.eval()
    predictions, labels_list = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions.extend(outputs.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    predictions = np.array(predictions).flatten()
    labels_array = np.array(labels_list).flatten()

    mse = mean_squared_error(labels_array, predictions)
    r_squared = r2_score(labels_array, predictions)

    return mse, r_squared



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# +
best_mse = float('inf')
best_params = {}

for lr in param_grid['learning_rate']:
    for epochs in param_grid['num_epochs']:
        for decay in param_grid['weight_decay']:
            # Initialize model
            model = BertRegressor()
            model.to(device)

            # Initialize optimizer with current hyperparameters
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

            # Training loop
            for epoch in range(epochs):
                model.train()
                for batch in train_loader:
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    loss = torch.nn.functional.mse_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()

            # Evaluation
            mse_eval, r_squared_eval = evaluate_model(model, eval_loader)
            print(f'LR: {lr}, Epochs: {epochs}, Weight Decay: {decay}, Eval MSE: {mse_eval}, Eval R-squared: {r_squared_eval}')


            # Update best model if current model is better
            if mse_eval < best_mse:
                best_mse = mse_eval
                best_params = {'learning_rate': lr, 'num_epochs': epochs, 'weight_decay': decay}
                # Save the best model
                torch.save(model.state_dict(), 'best_model.pth')

# Print best parameters
print(f'Best Parameters: {best_params}')

# -

model = BertRegressor()
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)

mse_test, r_squared_test = evaluate_model(model, test_loader)
print(f'Test MSE: {mse_test}')
print(f'Test R-squared: {r_squared_test}')


# +
model.eval()
all_predictions = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, _ = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        all_predictions.extend(outputs.cpu().numpy())
# -

predictions_scores = [float(pred) for pred in all_predictions]


predictions_df = pd.DataFrame(predictions_scores, columns=['predicted_score'])
predictions_csv_path = 'predicted_scores_gt_for_generative_all_model.csv'
predictions_df.to_csv(predictions_csv_path, index=False)


# +
def calculate_correlations(list1, list2):
    # Check if the lists are of the same length
    if len(list1) != len(list2):
         return "The lists are not of the same length"

    # Calculate Pearson correlation
    pearson_corr, pvaluep = scipy.stats.pearsonr(list1, list2)

    # Calculate Kendall Tau correlation
    kendall_corr, pvalue = scipy.stats.kendalltau(list1, list2)

    return pearson_corr, pvaluep, kendall_corr, pvalue

pearson_corr, pvaluep, kendall_corr, pvalue = calculate_correlations(test_scores_list, predictions_scores)
pearson_corr, pvaluep, kendall_corr, pvalue

# -

# pearson_corr, pvaluep, kendall_corr, pvalue = calculate_correlations(test_scores_list, predictions_scores)
# pearson_corr, pvaluep, kendall_corr, pvalue


# +
# model.eval()
# all_predictions = []

# with torch.no_grad():
#     for batch in train_loader:
#         input_ids, attention_mask, _ = [b.to(device) for b in batch]
#         outputs = model(input_ids, attention_mask=attention_mask)
#         all_predictions.extend(outputs.cpu().numpy())

# +
# with torch.no_grad():
#     for batch in eval_loader:
#         input_ids, attention_mask, _ = [b.to(device) for b in batch]
#         outputs = model(input_ids, attention_mask=attention_mask)
#         all_predictions.extend(outputs.cpu().numpy())

# +
# with torch.no_grad():
#     for batch in test_loader:
#         input_ids, attention_mask, _ = [b.to(device) for b in batch]
#         outputs = model(input_ids, attention_mask=attention_mask)
#         all_predictions.extend(outputs.cpu().numpy())

# +
# def write_embeddings(model, data_loader, device, set_name):
#     model.eval()  # Ensure the model is in evaluation mode.
#     embeddings = []  # List to store embeddings.

#     with torch.no_grad():  # No need to track gradients for this operation.
#         for batch in data_loader:
#             input_ids, attention_mask, _ = [b.to(device) for b in batch]  # We ignore labels here.
#             batch_embeddings = model.get_embeddings(input_ids, attention_mask)
#             embeddings.append(batch_embeddings.cpu().numpy())

#     embeddings = np.concatenate(embeddings, axis=0)  # Combine batch embeddings into a single array.

#     # Here you can write embeddings to a file or return them.
#     # For example, to save to a numpy file:
#     np.save(f"{set_name}_set_embeddings.npy", embeddings)
    
#     return embeddings


# +
# embeddings_train = write_embeddings(model, train_loader, device, 'train')
# embeddings_val = write_embeddings(model, eval_loader, device, 'val')
# embeddings_test = write_embeddings(model, test_loader, device, 'test')

# +
# embeddings = np.concatenate((embeddings_train, embeddings_val, embeddings_test), axis=0)

# +
# # Function to compute the best number of clusters using silhouette score
# def best_clusters_silhouette_and_plot(X, cluster_method, range_n_clusters=None, method_params={}):
#     best_score = -1
#     best_n_clusters = 0
#     scores = {}
#     if cluster_method == KMeans or cluster_method == AgglomerativeClustering:
#         for n_clusters in range_n_clusters:
#             model = cluster_method(n_clusters=n_clusters, **method_params)
#             cluster_labels = model.fit_predict(X)
#             silhouette_avg = silhouette_score(X, cluster_labels)
#             scores[n_clusters] = silhouette_avg
#             if silhouette_avg > best_score:
#                 best_score = silhouette_avg
#                 best_n_clusters = n_clusters
    
#     # Plotting the silhouette scores
#     plt.figure(figsize=(10, 6))
#     plt.plot(list(scores.keys()), list(scores.values()), marker='o', linestyle='-')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Silhouette score')
#     plt.title(f'Silhouette Scores for {cluster_method.__name__} Clustering')
#     plt.show()
    
#     return best_n_clusters, scores

# # DBSCAN doesn't require the number of clusters to be specified, so we search for the best parameters differently
# def best_params_dbscan(X, eps_values, min_samples_values):
#     best_score = -1
#     best_params = {}
#     for eps in eps_values:
#         for min_samples in min_samples_values:
#             db = DBSCAN(eps=eps, min_samples=min_samples)
#             cluster_labels = db.fit_predict(X)
#             # We calculate silhouette score only if there are more than 1 cluster (excluding noise)
#             if len(set(cluster_labels)) > 1:
#                 silhouette_avg = silhouette_score(X, cluster_labels)
#                 if silhouette_avg > best_score:
#                     best_score = silhouette_avg
#                     best_params = {'eps': eps, 'min_samples': min_samples}
#     return best_params, best_score


# -

# best_kmeans, _ = best_clusters_silhouette_and_plot(embeddings, KMeans, range_n_clusters=range(2, 25))


# +
# # t-SNE for 2D visualization
# tsne_2d = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne_2d.fit_transform(embeddings)
# print("tsne - 2d")
# # t-SNE for 3D visualization
# tsne_3d = TSNE(n_components=3, random_state=42)
# embeddings_3d = tsne_3d.fit_transform(embeddings)
# print("tsne - 3d")

# +
# #Write embeddings 
# np.save('2d_embeddings.npy',embeddings_2d)
# np.save('3d_embeddings.npy',embeddings_3d)
# embeddings_2d = np.load('2d_embeddings.npy')

# +
# from sklearn.preprocessing import MinMaxScaler
# import plotly.express as px
# import plotly.graph_objects as go

# def hex_to_rgb(hex):
#     hex = hex.lstrip('#')
#     return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

# plotly_colors_hex = px.colors.qualitative.D3
# plotly_colors_rgb = [hex_to_rgb(hex) for hex in plotly_colors_hex]

# +
# from sklearn.preprocessing import MinMaxScaler
# import plotly.io as pio
# new_min = 0.0
# new_max = 0.9


# train_inputs, train_masks, train_labels = tokenize_and_prepare(train_data)
# eval_inputs, eval_masks, eval_labels = tokenize_and_prepare(eval_data)
# test_inputs, test_masks, test_labels = tokenize_and_prepare(test_data)

# all_gts = [*train_labels,*eval_labels,*test_labels]
# all_gts = [ obj.item() for obj in all_gts]
# normalized_gts = new_min + ((all_gts - np.min(all_gts)) * (new_max - new_min) / (np.max(all_gts) - np.min(all_gts)))
# normalized_scores = new_min + ((all_predictions - np.min(all_predictions)) * (new_max - new_min) / (np.max(all_predictions) - np.min(all_predictions)))

# # Split the t-SNE results and scores by dataset
# train_tsne = embeddings_2d[:6000]
# val_tsne = embeddings_2d[6000:8000]
# test_tsne = embeddings_2d[8000:]


# # Define colors for each set
# color_train = 'blue'
# color_val = 'green'
# color_test = 'red'

# train_labels = normalized_gts[:6000]
# eval_labels = normalized_gts[6000:8000]
# test_labels = normalized_gts[8000:]

# # Define colors for each set
# color_train = 'blue'
# color_val = 'green'
# color_test = 'red'



# def create_trace(data, scores, name, color):
    
#     color_strings = []
#     for idx,score in enumerate(scores):
#         r,g,b = (255*score, 255*(1-score),0)
#         if score < 0.5:
#             # Transition from red to orange
#             # Red stays at 255, green goes from 0 to 165, blue stays at 0
#             r = 255
#             g = int(165 * (score * 2))  # Scale factor adjusted for half range [0, 0.5]
#             b = 0
#         else:
#             # Transition from orange to green
#             # Red decreases to 0, green goes from 165 to 255, blue stays at 0
#             r = int(255 * (1 - (score - 0.5) * 2))  # Adjust factor for range [0.5, 1]
#             g = 165 + int((255 - 165) * ((score - 0.5) * 2))  # Adjust green from 165 to 255
#             b = 0
    
#         color_string = f"rgba({r},{g},{b},1)"
#         color_strings.append(color_string)
    
#     return go.Scattergl(
#         x=data[:, 0], 
#         y=data[:, 1], 
#         mode='markers', 
#         marker=dict(
#             color=color_strings,
#             size=10,
#         ),
#         name=name
#     )

# # Ensure scores are flattened for opacity mapping
# train_trace = create_trace(train_tsne, train_labels, 'Train', plotly_colors_rgb[0])
# val_trace = create_trace(val_tsne, eval_labels, 'Validation', plotly_colors_rgb[0])
# test_trace = create_trace(test_tsne, test_labels, 'Test',plotly_colors_rgb[0])

# # Create a figure and add traces
# fig = go.Figure(data=[test_trace])

# # Update layout for a better visualization
# fig.update_layout(
#     title='t-SNE visualization of Fine-Tuned BERT (HBPP) -  Color Coded Difficulty',
#     xaxis_title='Component 1',
#     yaxis_title='Component 2',
#     legend_title='Dataset',
#     height=900,
#     width=1600,
#     font = dict(size=25),
#     xaxis=dict(
#         tickfont=dict(size=35)  # Adjust the x-axis tick font size here
#     ),
#     yaxis=dict(
#         tickfont=dict(size=35)  # Adjust the y-axis tick font size here
#     )       
# )

# # Show the figure
# fig.show()
# pio.write_image(fig, 'tsne_visualization.pdf')

# +
# pio.write_image(fig, 'tsne_visualization.pdf')

# +
# pip install -U kaleido

# +
# import importlib
# import kaleido
# importlib.reload(kaleido)

# +
# importlib.reload(pio)

# +
# test_scores_list, predictions_score

# +
# all_gts = [*train_labels,*eval_labels,*test_labels]
# all_gts = [ obj.item() for obj in all_gts]

# all_preds = [item.item() for item in all_predictions]

# +
# len(all_preds)

# +
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px

# # Assuming `all_gts` and `all_preds` are already defined lists or arrays with appropriate sizes
# # For demonstration purposes, I'll simulate these arrays with random data
# import numpy as np

# np.random.seed(0)  # For reproducibility

# # Simulate data


# # Create a DataFrame with the specified slices
# df = pd.DataFrame({
#     'ground_truth': all_gts[8000:10000],
#     'prediction': all_preds[8000:10000]
# })

# #fig = px.density_contour(df, x="ground_truth", y="prediction", height=600, marginal_x='histogram', marginal_y='histogram')

# #fig.update_traces(contours_coloring="fill", contours_showlabels = True)

# # Update layout for a better visualization


# # Create a Histogram2dContour plot
# fig = go.Figure(go.Histogram2dContour(
#     x=df['ground_truth'],
#     y=df['prediction'],
#     colorscale='Reds',  # You can choose other color scales such as 'Blues', 'Greens', 'Jet', etc.
#     contours=dict(
#         showlabels=True,  # Show labels on contours
#         labelfont=dict(  # Font settings for labels
#             size=12,
#             color='white',
#         )
#     )
# ))

# # Optionally, add scatter plot points on top
# fig.add_trace(go.Scatter(
#     x=df['ground_truth'],
#     y=df['prediction'],
#     mode='markers',
#     marker=dict(
#         color='red',
#         size=3,
#         opacity=1
#     ),
#     showlegend=False
# ))

# fig.update_layout(
#     xaxis_title="Ground Truth",
#     yaxis_title="Prediction Difficulty",
#     height=900,
#     width=1600,
#     title='Density Contour of Predictions vs. Ground Truth - Finetuned Bert - GT',
#     font=dict(size=25)
# )




# fig.show()
# pio.write_image(fig, 'density_contour_finetuned_bert_gt.pdf')

# +
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px

# # Assuming `all_gts` and `all_preds` are already defined lists or arrays with appropriate sizes
# # For demonstration purposes, I'll simulate these arrays with random data
# import numpy as np

# np.random.seed(0)  # For reproducibility

# # Simulate data


# # Create a DataFrame with the specified slices
# df = pd.DataFrame({
#     'ground_truth': all_gts[8000:10000],
#     'prediction': all_preds[8000:10000]
# })

# fig = px.box(df, x="ground_truth", y="prediction", title='Boxplot of Predictions and Ground Truth - Finetuned Bert - GT', height=600)
# #plt.xlabel('Type')
# #plt.ylabel('Values')

# # Update layout for a better visualization
# fig.update_layout(
#     xaxis_title="Ground Truth",
#     yaxis_title="Difficulty Prediction",
#     height=900,
#     width=1600,
#     font = dict(size=25)
# )



# fig.show()
# pio.write_image(fig, 'boxplit_finetuned_bert_gt.pdf')
