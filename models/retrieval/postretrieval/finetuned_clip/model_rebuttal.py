import torch
import pickle
import pandas as pd

# -- Step 1: Load datasets --
train_dataset = pickle.load(open("./train_dataset_rebuttal.pickle", "rb"))
validation_dataset = pickle.load(open("./val_dataset_rebuttal.pickle", "rb"))
test_dataset = pickle.load(open("./test_dataset_rebuttal.pickle", "rb"))

selected_model = "blip2"
train_source = "mscoco"
test_source = "mscoco"

train_retrieval_gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\{selected_model}\{selected_model}_retrieval_train_results.csv"
val_retrieval_gt_path  = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\{selected_model}\{selected_model}_retrieval_val_results.csv"
test_retrieval_gt_path  = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\{selected_model}\{selected_model}_retrieval_test_results.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"


# -- Step 2: Dataset definition --
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, gt_path, accepted_source):
        # Read the ground truth file
        gt_file = pd.read_csv(gt_path)
        # Filter by source
        indices = gt_file[gt_file["source"] == accepted_source].index.tolist()
        
        self.dataset = []
        for i in range(len(dataset)):
            # The original code uses i // 50 in indices to filter
            if (i // 50) in indices:
                self.dataset.append(dataset[i])

    def __getitem__(self, index):
        combined_features, individual_score = self.dataset[index]
        x = torch.tensor(combined_features, dtype=torch.float).to(device)
        y = torch.tensor(individual_score, dtype=torch.float).to(device)
        return x, y

    def __len__(self):
        return len(self.dataset)


# -- Step 3: DataLoaders --
batch_size = 256

train_loader = torch.utils.data.DataLoader(
    CustomDataset(train_dataset, train_retrieval_gt_path, train_source),
    batch_size=batch_size,
    shuffle=True,
)

validation_loader = torch.utils.data.DataLoader(
    CustomDataset(validation_dataset, val_retrieval_gt_path, train_source),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    CustomDataset(test_dataset, test_retrieval_gt_path, test_source),
    batch_size=batch_size,
    shuffle=False,
)


# -- Step 4: Define the Model (no final sigmoid for regression) --
class NeuralNetworkRegressor(torch.nn.Module):
    def __init__(self):
        super(NeuralNetworkRegressor, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)  # for regularization

    def forward(self, x):
        # x expected to be shape: [batch_size, 2, 512] => flatten to [batch_size, 1024]
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        # final layer for regression => no sigmoid
        x = self.fc3(x)
        return x.squeeze(1)  # shape: [batch_size]


# -- Step 5: Hyperparameter Grid Search --
hyperparameters = {
    "learning_rate": [1e-5, 1e-4],
    "weight_decay": [0.0, 0.01, 0.1],
}

best_val_loss = float("inf")
best_hyperparams = {}

num_epochs = 25

for lr in hyperparameters["learning_rate"]:
    for decay in hyperparameters["weight_decay"]:
        
        # Initialize model, loss, optimizer, scheduler
        model = NeuralNetworkRegressor().to(device)
        loss_fn = torch.nn.SmoothL1Loss(beta=0.1)  # Huber loss with custom beta
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        for epoch in range(num_epochs):
            # ---- Training ----
            model.train()
            total_train_loss = 0.0
            for combined_features, individual_score in train_loader:
                # Forward
                preds = model(combined_features)
                loss = loss_fn(preds, individual_score)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()

            # ---- Validation ----
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for combined_features_val, individual_score_val in validation_loader:
                    preds_val = model(combined_features_val)
                    loss_val = loss_fn(preds_val, individual_score_val)
                    val_loss += loss_val.item()

            val_loss /= len(validation_loader)
            total_train_loss /= len(train_loader)

            # Step the scheduler with validation loss
            scheduler.step(val_loss)

            # Check if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hyperparams = {"learning_rate": lr, "weight_decay": decay}
                print(f"New best model => Val Loss: {best_val_loss:.6f} (lr={lr}, wd={decay})")
                torch.save(model.state_dict(), f"./model_{train_source}_rebuttal.pt")

            print(
                f"Epoch [{epoch+1}/{num_epochs}]"
                f" | Train Loss: {total_train_loss:.6f}"
                f" | Val Loss: {val_loss:.6f}"
                f" | LR: {lr}"
                f" | WD: {decay}"
            )


# -- Step 6: Evaluation on Test Set --
model = NeuralNetworkRegressor().to(device)
model.load_state_dict(torch.load(f"./model_{train_source}_rebuttal.pt"))
model.eval()

test_predictions = []
with torch.no_grad():
    for combined_features_test, individual_score_test in test_loader:
        preds_test = model(combined_features_test)
        test_predictions.extend(preds_test.tolist())

# Save test predictions
pickle.dump(
    test_predictions,
    open(f"./rebuttal_unprocessed_{selected_model}_{train_source}_predictions_2.pickle", "wb")
)

print("Best Hyperparameters:", best_hyperparams)
print("Best Validation Loss:", best_val_loss)
