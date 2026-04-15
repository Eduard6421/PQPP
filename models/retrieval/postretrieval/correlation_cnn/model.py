
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


metric = "reciprocal_rank"
selected_model = "clip"
other_model = "blip2"
matrix_index = {
    "clip": 0,
    "blip2": 1,
}


train_retrieval_gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\{selected_model}\{selected_model}_retrieval_train_results.csv"
val_retrieval_gt_path  = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\{selected_model}\{selected_model}_retrieval_val_results.csv"
test_retrieval_gt_path  = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\{selected_model}\{selected_model}_retrieval_test_results.csv"

ground_truth_train = pd.read_csv(train_retrieval_gt_path)
ground_truth_val= pd.read_csv(val_retrieval_gt_path)
ground_truth_test = pd.read_csv(test_retrieval_gt_path)



train_array_scores = np.array(ground_truth_train[metric])
validation_array_scores = np.array(ground_truth_val[metric])
test_array_scores = np.array(ground_truth_test[metric])


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, pickle_file, scores):
        with open(pickle_file, "rb") as f:
            self.data = pickle.load(f)
            self.scores = scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, _ = self.data[idx]
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        image = torch.tensor(image[matrix_index[selected_model]], dtype=torch.float32)
        return image, score
    


# DataLoader setup
train_dataset = CustomDataset("corr_train_dataset.pickle", train_array_scores)
validation_dataset = CustomDataset(
    "corr_val_dataset.pickle", validation_array_scores
)
test_dataset = CustomDataset("corr_test_dataset.pickle", test_array_scores)



class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)

        self.sigmoid = nn.Sigmoid()

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers

        self.fc1 = nn.Linear(
            64 * 32 * 32, 1024
        )  # Adjust the input size to match the output of the last conv layer
        self.fc2 = nn.Linear(1024, 1)  # Output layer for regression (1 output value)

    def forward(self, x):
        # Apply convolutional layers with ReLU and pooling
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the output for the fully connected layer
        x = x.view(
            -1, 64 * 32 * 32
        )  # Adjust the size to match the output of the last conv layer

        # Fully connected layers with ReLU activation for the first
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# Hyperparameters
hyperparameters = {
    "learning_rate": [1e-3,1e-4,1e-5],
    "num_epochs": [25],
    "weight_decay": [0, 0.01],
}

# Hyperparameter search
best_val_loss = float("inf")
best_hyperparams = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("starting training")

for lr in hyperparameters["learning_rate"]:
    for epochs in hyperparameters["num_epochs"]:
        for wd in hyperparameters["weight_decay"]:
            # Model initialization
            model = CNNRegressor().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            # Training and validation loop
            for epoch in range(epochs):
                model.train()
                for images, scores in train_loader:
                    images, scores = images.to(device), scores.to(device)
                    optimizer.zero_grad()

                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), scores)
                    loss.backward()
                    
                    optimizer.step()


                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for images, scores in validation_loader:
                        images, scores = images.to(device), scores.to(device)
                        outputs = model(images)
                        val_loss += criterion(outputs.squeeze(), scores).item()

                val_loss /= len(validation_loader)

                # Print current epoch
                print(
                    f"Epoch: {epoch+1}/{epochs}, Val Loss: {val_loss}, lr: {lr}, wd: {wd}"
                )

                # Save the best model and hyperparameters
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_hyperparams = {
                        "learning_rate": lr,
                        "num_epochs": epochs,
                        "weight_decay": wd,
                    }
                    torch.save(model.state_dict(), f"{selected_model}_{metric}_best_model.pth")
                    print(
                        f"New best model saved with val_loss: {val_loss}, hyperparameters: {best_hyperparams}"
                    )

# Output best hyperparameters after search
print(f"Best Hyperparameters: {best_hyperparams}")
model.load_state_dict(torch.load(f"{selected_model}_{metric}_best_model.pth"))
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions.extend(outputs.squeeze().tolist())

with open(f"{selected_model}_test_predictions_{metric}.pickle", "wb") as f:
    pickle.dump(predictions, f)
