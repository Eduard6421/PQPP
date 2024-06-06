from collections import Counter
import pickle

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers import evaluation, util
import random
import os

dataset_path = "../dataset/sentence_transformer_dataset_new.pickle"


def generate_dataset(dataset_path):
    # Load pickle
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    full_dataset = []
    all_indexes = set()

    for item in dataset:
        (query_index, query_text, retrieval_text, label) = item
        full_dataset.append(
            (
                query_index,
                InputExample(
                    texts=[query_text, retrieval_text],
                    label=(0.0 if label == False else 1.0),
                ),
            )
        )
        all_indexes.add(query_index)

    all_indexes = list(all_indexes)

    # Randomize the order of indexes
    random_indexes = random.sample(all_indexes, len(all_indexes))

    # Calculate the sizes of each split
    total_size = len(random_indexes)
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size

    # Split the indexes into train, validation, and test sets
    train_indexes = random_indexes[:train_size]
    val_indexes = random_indexes[train_size : train_size + val_size]
    test_indexes = random_indexes[train_size + val_size :]

    # Select the examples for each split

    train_examples = []
    val_examples = []
    test_examples = []

    for item in full_dataset:
        (query_index, example) = item
        if query_index in train_indexes:
            train_examples.append(example)
        elif query_index in val_indexes:
            val_examples.append(example)
        elif query_index in test_indexes:
            test_examples.append(example)

    # Print length of each and appropiate message
    print("Length of train_examples:", len(train_examples))
    print("Length of val_examples:", len(val_examples))
    print("Length of test_examples:", len(test_examples))

    # Save the datasets
    with open("../dataset/train_examples_new.pickle", "wb") as f:
        pickle.dump(train_examples, f)

    with open("../dataset/val_examples_new.pickle", "wb") as f:
        pickle.dump(val_examples, f)

    with open("../dataset/test_examples_new.pickle", "wb") as f:
        pickle.dump(test_examples, f)


def check_label_distribution(dataset_path):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    cnt = Counter()
    for item in dataset:
        cnt[item.label] += 1


# check_label_distribution("../dataset/train_examples.pickle")
# check_label_distribution("../dataset/val_examples.pickle")
# check_label_distribution("../dataset/test_examples.pickle")
generate_dataset(dataset_path)
