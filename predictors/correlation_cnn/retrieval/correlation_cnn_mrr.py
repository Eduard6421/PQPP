import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


gt_all_mrr_scores = "../../../dataset/retrieval_annotation_scores/avg_scores_rr.pickle"
with open(gt_all_mrr_scores, "rb") as f:
    gt_all_mrr_scores = pickle.load(f)
parsed_scores = []
for i in range(10000):
    parsed_scores.append(gt_all_mrr_scores[i])
train_array_scores = parsed_scores[:6000]
validation_array_scores = parsed_scores[6000:8000]
test_array_scores = parsed_scores[8000:10000]


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
        image = torch.tensor(image, dtype=torch.float)
        score = self.scores[idx]
        return image.float(), torch.tensor(score, dtype=torch.float)


class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(
            256 * 128 * 128, 1024
        )  # Adjust the input size to match the output of the last conv layer
        self.fc2 = nn.Linear(1024, 1)  # Output layer for regression (1 output value)

    def forward(self, x):
        # Apply convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten the output for the fully connected layer
        x = x.view(
            -1, 256 * 128 * 128
        )  # Adjust the size to match the output of the last conv layer

        # Fully connected layers with ReLU activation for the first
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation, as this is a regression task

        return x


# DataLoader setup
train_dataset = CustomDataset("train_data_mrr.pickle", train_array_scores)
validation_dataset = CustomDataset(
    "validation_data_mrr.pickle", validation_array_scores
)
test_dataset = CustomDataset("test_data_mrr.pickle", test_array_scores)

# Hyperparameters
hyperparameters = {
    "learning_rate": [1e-5, 1e-4, 5e-5],
    "num_epochs": [25],
    "weight_decay": [0, 0.1, 0.01],
}

# Hyperparameter search
best_val_loss = float("inf")
best_hyperparams = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
                    torch.save(model.state_dict(), "best_model_mrr.pth")
                    print(
                        f"New best model saved with val_loss: {val_loss}, hyperparameters: {best_hyperparams}"
                    )
model = CNNRegressor().to(device)

# Output best hyperparameters after search
print(f"Best Hyperparameters: {best_hyperparams}")
model.load_state_dict(torch.load("best_model_mrr.pth"))
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions.extend(outputs.squeeze().tolist())

# Save predictions to a pickle file
with open("test_predictions_mrr.pickle", "wb") as f:
    pickle.dump(predictions, f)
