import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from torchvision import transforms
import pandas as pd

selected_model = "clip"


train_data_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{selected_model}\{selected_model}_train.csv"
val_data_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{selected_model}\{selected_model}_val.csv"
test_data_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{selected_model}\{selected_model}_test.csv"


train_source = "drawbench"
test_source = "mscoco"

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, pickle_file, score_file_path, accepted_source ):

        scores = pd.read_csv(score_file_path)

        # get the indexes of scores that have source in accepted_sources
        indices = scores[scores['source'] == accepted_source].index.tolist()

        data = pickle.load(open(pickle_file, "rb"))

        self.data = [data[i] for i in indices]
        self.scores = scores.iloc[indices]['score'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, _ = self.data[idx]
        score = self.scores[idx]
        score = (score + 1) / 3
        return torch.tensor(image,dtype=torch.float), torch.tensor(score, dtype=torch.float)


class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        
        # Define convolutional layers individually
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Define fully connected layers individually
        self.fc1 = nn.Linear(64 * 64 * 64, 512)  # Adjust this based on final conv layer output size
        self.fc_relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 1)
        self.fc_relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through each layer explicitly, with shape printing

        x = x.unsqueeze(1)

        #print("Input shape:", x.shape)
        x = self.conv1(x)
        #print("After conv1:", x.shape)
        
        x = self.relu1(x)
        #print("After relu1:", x.shape)
        
        x = self.pool1(x)
        #print("After pool1:", x.shape)
        x = self.conv2(x)
        #print("After conv2:", x.shape)
        x = self.relu2(x)
        #print("After relu2:", x.shape)
        x = self.pool2(x)
        #print("After pool2:", x.shape)
        x = self.conv3(x)
        #print("After conv3:", x.shape)
        x = self.relu3(x)
        #print("After relu3:", x.shape)
        x = self.pool3(x)
        #print("After pool3:", x.shape)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        #print("After flattening:", x.shape)
        
        x = self.fc1(x)
        #print("After fc1:", x.shape)
        x = self.fc_relu1(x)
        #print("After fc_relu1:", x.shape)
        x = self.dropout(x)
        #print("After dropout:", x.shape)
        x = self.fc2(x)
        #print("After fc2:", x.shape)
        x = self.sigmoid(x)
        #print("After fc_relu2:", x.shape)
        
        return x

# DataLoader setup
train_dataset = CustomDataset(fr"correlation_cnn_train_data.pkl", train_data_path, train_source)
validation_dataset = CustomDataset(fr"correlation_cnn_val_data.pkl", val_data_path,train_source)
test_dataset = CustomDataset(fr"correlation_cnn_test_data.pkl",test_data_path,test_source)

# Hyperparameters
hyperparameters = {
    "learning_rate": [1e-3],
    "num_epochs": [25],
    "weight_decay": [0, 0.1,],
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
                    torch.save(model.state_dict(),fr"crossmodel_{train_source}_trained_{selected_model}_best_model.pth")
                    print(
                        f"New best model saved with val_loss: {val_loss}, hyperparameters: {best_hyperparams}"
                    )

    # Output best hyperparameters after search
    print(f"Best Hyperparameters: {best_hyperparams}")
model.load_state_dict(torch.load(fr"crossmodel_{train_source}_trained_{selected_model}_best_model_new.pth"))
model.eval()
predictions = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions.extend(outputs.squeeze().tolist())

# Save predictions to a pickle file
with open(fr"crossmodel_{train_source}_trained_{selected_model}_test_predictions_new.pickle", "wb") as f:
    pickle.dump(predictions, f)
