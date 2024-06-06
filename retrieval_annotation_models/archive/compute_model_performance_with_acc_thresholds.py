import itertools
import pickle
import random
import os
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers import evaluation, util
from sklearn.metrics import roc_curve, auc
from collections import Counter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve


def cosine_similarity(embeddings1, embeddings2):
    return np.dot(embeddings1, embeddings2) / (
        np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
    )


dataset_path = "../dataset/sentence_transformer_dataset.pickle"

from sklearn.metrics import accuracy_score


def compute_accuracy_threshold(true_labels, predictions):
    """
    Compute the best threshold for maximizing the overall accuracy of the model using actual values from predictions.

    :param true_labels: Array of true labels (0 or 1).
    :param predictions: Array of probabilities or logits for label 1 predicted by the model.
    :return: Best threshold and its corresponding accuracy.
    """
    best_accuracy = 0
    best_threshold = 0.0  # Default threshold

    # Extract unique thresholds from the predictions
    unique_thresholds = np.unique(predictions)

    # Iterate over the actual values from the predictions
    for threshold in unique_thresholds:
        # Apply threshold to predictions to get binary classification
        predicted_labels = (predictions >= threshold).astype(int)
        # Calculate accuracy for the current threshold
        accuracy = accuracy_score(true_labels, predicted_labels)

        # Update the best accuracy and threshold if current accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold


def parse_groundtruth(evaluation_type="val"):
    with open(f"../dataset/{evaluation_type}_examples_new.pickle", "rb") as f:
        data = pickle.load(f)
        sentences_1 = []
        sentences_2 = []
        true_labels = []
        for item in data:
            sentences_1.append(item.texts[0])
            sentences_2.append(item.texts[1])
            true_labels.append(item.label)
        return sentences_1, sentences_2, true_labels


def evaluate_sentence_transformer(path, evaluation_type="val", best_threshold=None):
    # "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(path)

    sentences_1, sentences_2, true_labels = parse_groundtruth(evaluation_type)

    # Compute embeddings
    embeddings1 = model.encode(sentences_1, convert_to_tensor=False)
    embeddings2 = model.encode(sentences_2, convert_to_tensor=False)

    predictions = [
        cosine_similarity(embeddings1[i], embeddings2[i])
        for i in range(len(true_labels))
    ]

    if best_threshold is None:
        print("Threshold Not computed")
        best_threshold = compute_accuracy_threshold(true_labels, predictions)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Compute the predictions with a treshold of 0.85
    predictions = [1 if x >= best_threshold else 0 for x in predictions]

    # Compute the accuracy score
    correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predictions[i]:
            correct += 1

    # Compute f1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions)
    f1_score_class_1 = f1[1]

    print("Evaluation Suite:", evaluation_type)
    print("Accuracy:", correct / len(true_labels))
    print("AUC Score:", roc_auc)
    print("F1 Score (Positive Labels):", f1_score_class_1)

    return best_threshold


def evaluation_suite():
    paths = (
        "./models/lr=0.0001_decay=0_opt=AdamW_sched=warmupcosine",
        "./models/lr=0.0001_decay=0_opt=AdamW_sched=warmupcosinewithhardrestarts",
        "./models/lr=0.0001_decay=0.01_opt=AdamW_sched=warmupcosine",
        "./models/lr=0.0001_decay=0.01_opt=AdamW_sched=warmupcosinewithhardrestarts",
        "./models/lr=1e-05_decay=0_opt=AdamW_sched=warmupcosine",
        "./models/lr=1e-05_decay=0_opt=AdamW_sched=warmupcosinewithhardrestarts",
        "./models/lr=1e-05_decay=0.01_opt=AdamW_sched=warmupcosine",
        "./models/lr=1e-05_decay=0.01_opt=AdamW_sched=warmupcosinewithhardrestarts",
        "./models/lr=1e-06_decay=0_opt=AdamW_sched=warmupcosine",
        "./models/lr=1e-06_decay=0_opt=AdamW_sched=warmupcosinewithhardrestarts",
        "./models/lr=1e-06_decay=0.01_opt=AdamW_sched=warmupcosine",
        "./models/lr=1e-06_decay=0.01_opt=AdamW_sched=warmupcosinewithhardrestarts",
    )

    for path in paths:
        print("Model type" + path)
        best_treshold = evaluate_sentence_transformer(path, "val")
        evaluate_sentence_transformer(path, "test", best_treshold)


print("Model type: Pretrained")
best_threshold = evaluate_sentence_transformer(
    "sentence-transformers/all-mpnet-base-v2", "val"
)
evaluate_sentence_transformer(
    "sentence-transformers/all-mpnet-base-v2", "test", best_threshold=best_threshold
)
evaluation_suite()
