import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from torchvision import transforms


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, "rb") as f:
            self.data = pickle.load(f)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, score = self.data[idx]
        image = self.transform(image)
        return image.float(), torch.tensor(score, dtype=torch.float)


class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.conv_layers = nn.Sequential(
            # Adjust the first layer to accept 1 input channel
            nn.Conv2d(1, 16, 3, 1, 1),  # Changed from 3 input channels to 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            # Adjust the size if the spatial dimensions change due to no pooling or convolution layers being modified
            nn.Linear(
                64 * 64 * 64, 512
            ),  # Size may need to be recalculated if the architecture changes
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Adding dropout for regularization
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc_layers(x)
        return x


# DataLoader setup
train_dataset = CustomDataset("train_data.pickle")
validation_dataset = CustomDataset("validation_data.pickle")
test_dataset = CustomDataset("test_data.pickle")

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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

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
                    torch.save(model.state_dict(), "best_model.pth")
                    print(
                        f"New best model saved with val_loss: {val_loss}, hyperparameters: {best_hyperparams}"
                    )

    # Output best hyperparameters after search
    print(f"Best Hyperparameters: {best_hyperparams}")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions.extend(outputs.squeeze().tolist())

# Save predictions to a pickle file
with open("test_predictions.pickle", "wb") as f:
    pickle.dump(predictions, f)
