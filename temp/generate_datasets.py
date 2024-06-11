import pandas as pd

# Load the dataset
df = pd.read_csv("../dataset/dataset.csv")

# Split the dataset
train_df = df.iloc[:6000]
validation_df = df.iloc[6000:8000]
test_df = df.iloc[8000:10000]

# Write the datasets to CSV files
train_df.to_csv("../dataset/train.csv", index=False)
validation_df.to_csv("../dataset/validation.csv", index=False)
test_df.to_csv("../dataset/test.csv", index=False)
