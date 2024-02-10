import torch
import pickle

# import train / validation and test datasets
train_dataset = pickle.load(open("./train_dataset.pickle", "rb"))
validation_dataset = pickle.load(open("./validation_dataset.pickle", "rb"))
test_dataset = pickle.load(open("./test_dataset.pickle", "rb"))


# The strucutre of the elements in the array are base_image_id, combined_features, individual_scor
# We need to create a pytorch dataset that takes this


device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        combined_features, individual_score = self.dataset[index]
        return (
            torch.tensor(combined_features, dtype=torch.float).to(device),
            torch.tensor(individual_score, dtype=torch.float).to(device),
        )

    def __len__(self):
        return len(self.dataset)


# Create a dataloader for the train, validation and test datasets, with batch size 32

train_loader = torch.utils.data.DataLoader(
    CustomDataset(train_dataset), batch_size=256, shuffle=True
)

validation_loader = torch.utils.data.DataLoader(
    CustomDataset(validation_dataset), batch_size=256, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    CustomDataset(test_dataset), batch_size=256, shuffle=False
)


# Create a Neural network that takes the combined features and outputs a score (combined features have a size of 2x512)


class NeuralNetworkClassifier(torch.nn.Module):
    def __init__(self):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # Input has size 2x512 so we need to reshape
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Write a training loop that saves the best model overall.
# It should do a grid search for learning_rate 1e-5, 5e-5 1e-4 and weight_decay 0, 0.01, 0.1

# Hyperparameters
hyperparameters = {
    "learning_rate": [1e-5, 1e-4, 5e-5],
    "weight_decay": [0, 0.1, 0.01],
}

# Hyperparameter search
best_val_loss = float("inf")
best_hyperparams = {}


num_epochs = 25
for lr in hyperparameters["learning_rate"]:
    for decay in hyperparameters["weight_decay"]:
        # Create a model
        model = NeuralNetworkClassifier().to(device)
        # Create a loss function
        loss_fn = torch.nn.BCELoss()
        # Create an optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

        # Training and validation loop
        for epoch in range(num_epochs):
            model.train()
            for combined_features, individual_score in train_loader:
                # Forward pass
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
                torch.save(model.state_dict(), "./best_model.pt")
            print("Epoch: ", epoch, "Val loss: ", val_loss, "lr: ", lr, "wd: ", decay)


# Evaluate the best model on the test set and save the regression output to test_predictions.pickle
model = NeuralNetworkClassifier().to(device)
model.load_state_dict(torch.load("./best_model.pt"))
model.eval()

test_predictions = []
with torch.no_grad():
    for combined_features, individual_score in test_loader:
        combined_features = combined_features.squeeze(1)
        pred = model(combined_features).squeeze(1)
        test_predictions.extend(pred.tolist())
pickle.dump(test_predictions, open("./test_predictions.pickle", "wb"))
