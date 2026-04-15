import pandas as pd
import gensim.downloader as api
import networkx as nx
import numpy as np
import networkx as nx
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import pickle
import os
import sys

# Dynamically add the parent directory of `retrieval_process` to `sys.path`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Now import `longclip` from its location within `retrieval_process`
from postgenerative.LongClip.model import longclip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = longclip.load(fr"C:\Users\User\Desktop\Research\PQPP\models\generative\postgenerative\LongClip\checkpoints\longclip-B.pt", device=device)


selected_model = "glide"
train_source = "mscoco"
test_source = "mscoco"


train_data_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{selected_model}\{selected_model}_train.csv"
val_data_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{selected_model}\{selected_model}_val.csv"
test_data_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{selected_model}\{selected_model}_test.csv"



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, gt_path,other_gt_path, accepted_source):

        loaded_dataset = dataset
        ground_truth = pd.read_csv(gt_path)
        other_gt = pd.read_csv(other_gt_path)
        new_scores = other_gt[fr"{selected_model}_new"].tolist()
        # get indexes from ground_truth where source is accepted_source
        indices = ground_truth[ground_truth['source'] == accepted_source].index.tolist()

        # in order to get the real index of an items we need to divide the position of the item by 4 as we have groups of four.
        # i want you to keep in the dataste only from the indexes that are in the indices list

        self.dataset = []

        last_gt_selected = 0

        for i in range(len(loaded_dataset)):
            if i // 4 in indices:
                real_gt_score = new_scores[last_gt_selected]
                base_image_id, combined_features, _individual_score = loaded_dataset[i]
                self.dataset.append((base_image_id, combined_features, real_gt_score))

                if(i%4 == 3):
                    last_gt_selected += 1
        print(last_gt_selected)

    def __getitem__(self, index):
        base_image_id, combined_features, individual_score = self.dataset[index]
        individual_score = individual_score
        return (
            base_image_id,
            torch.tensor(combined_features, dtype=torch.float).to(device),
            torch.tensor(individual_score, dtype=torch.float).to(device),
        )

    def __len__(self):
        return len(self.dataset)

# import train / validation and test datasets
train_dataset = pickle.load(open(fr"./{selected_model}_train_pairs.pkl", "rb"))
validation_dataset = pickle.load(open(fr"./{selected_model}_val_pairs.pkl", "rb"))
test_dataset = pickle.load(open(fr"./{selected_model}_test_pairs.pkl", "rb") )



train_loader = torch.utils.data.DataLoader(
    CustomDataset(train_dataset, train_data_path, fr"C:\Users\User\Desktop\Research\PQPP\rebuttal\automatic_train_gt.csv", train_source), batch_size=256, shuffle=True
)

validation_loader = torch.utils.data.DataLoader(
    CustomDataset(validation_dataset, val_data_path,fr"C:\Users\User\Desktop\Research\PQPP\rebuttal\automatic_val_gt.csv",train_source), batch_size=256, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    CustomDataset(test_dataset,test_data_path,fr"C:\Users\User\Desktop\Research\PQPP\rebuttal\automatic_test_gt.csv", test_source), batch_size=32, shuffle=False
)



class NeuralNetworkRegressor(torch.nn.Module):
    def __init__(self):
        super(NeuralNetworkRegressor, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        # Input has size 2x512 so we need to reshape
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Hyperparameters
hyperparameters = {
    "learning_rate": [1e-5, 1e-4, 5e-5],
    "weight_decay": [0, 0.1, 0.01],
}
best_val_loss = float("inf")
num_epochs = 25
for lr in hyperparameters["learning_rate"]:
    for decay in hyperparameters["weight_decay"]:
        # Create a model
        model = NeuralNetworkRegressor().to(device)
        # Create a loss function
        loss_fn = torch.nn.MSELoss()
        # Create an optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        # Training and validation loop
        for epoch in range(num_epochs):
            model.train()
            for base_image_id, combined_features, individual_score in train_loader:
                # Forward pass]
                combined_features = combined_features.squeeze(1)

                pred = model(combined_features).squeeze(1)

                loss = loss_fn(pred, individual_score)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for (
                    base_image_id,
                    combined_features,
                    individual_score,
                ) in validation_loader:
                    combined_features = combined_features.squeeze(1)
                    pred = model(combined_features).squeeze(1)
                    val_loss += loss_fn(pred, individual_score).item()

            val_loss /= len(validation_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyperparams = {"learning_rate": lr, "weight_decay": decay}
                print(
                    f"Saving best model with val loss: {best_val_loss}, learing_rate {lr}, weight_decay {decay}"
                )
                torch.save(model.state_dict(), fr"./rebuttal_model_{train_source}_{selected_model}_best_model.pt")
            print("Epoch: ", epoch, "Val loss: ", val_loss, "lr: ", lr, "wd: ", decay)
            # Evaluate the best model on the test set and save the regression output to test_predictions.pickle
model = NeuralNetworkRegressor().to(device)
model.load_state_dict(torch.load(fr"./rebuttal_model_{train_source}_{selected_model}_best_model.pt"))
model.eval()

test_predictions = []
with torch.no_grad():
    for base_image_id, combined_features, individual_score in test_loader:
        combined_features = combined_features.squeeze(1)
        pred = model(combined_features).squeeze(1)
        test_predictions.extend(pred.tolist())


pickle.dump(test_predictions, open(fr"./rebuttal_model_{train_source}_{selected_model}_test_predictions.pickle", "wb"))
