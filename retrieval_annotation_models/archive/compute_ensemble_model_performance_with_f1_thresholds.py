import itertools
import pickle
import random
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers import evaluation, util
from sklearn.metrics import roc_auc_score, roc_curve, auc
from collections import Counter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from scipy.stats import mode


def cosine_similarity(embeddings1, embeddings2):
    return np.dot(embeddings1, embeddings2) / (
        np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
    )


paths = [
    "./models/lr=0.0001_decay=0_opt=AdamW_sched=warmupcosine",
    "./models/lr=0.0001_decay=0.01_opt=AdamW_sched=warmupcosine",
    "./models/lr=1e-06_decay=0.01_opt=AdamW_sched=warmupcosinewithhardrestarts",
]


def compute_f1_threhsold_for_label1(true_labels, predictions):
    precisions, recalls, thresholds = precision_recall_curve(
        true_labels, predictions, pos_label=1
    )

    best_f1 = 0
    best_threshold = 0
    # Iterate through all the thresholds to find the best one
    for i in range(len(thresholds)):
        precision = precisions[i + 1]  # Shift index for precision and recall
        recall = recalls[i + 1]
        # Check to ensure we're not dividing by zero
        if (precision + recall) > 0:
            # Calculate F1 score for class 1
            f1 = 2 * (precision * recall) / (precision + recall)
            # Check if this F1 score is the best one
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresholds[i]

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


def get_model_predictions(model, evaluation_type):
    sentences_1, sentences_2, true_labels = parse_groundtruth(evaluation_type)
    # Compute embeddings
    embeddings1 = model.encode(sentences_1, convert_to_tensor=False)
    embeddings2 = model.encode(sentences_2, convert_to_tensor=False)

    predictions = [
        cosine_similarity(embeddings1[i], embeddings2[i])
        for i in range(len(true_labels))
    ]

    best_threshold = compute_f1_threhsold_for_label1(true_labels, predictions)

    new_predictions = [
        cosine_similarity(embeddings1[i], embeddings2[i])
        for i in range(len(true_labels))
    ]

    return new_predictions, best_threshold


def soft_labels_ensemble(paths):
    val_predictions = []
    test_predictions = []

    _, _, val_true_labels = parse_groundtruth("val")
    _, _, test_true_labels = parse_groundtruth("test")

    for path in paths:
        model = SentenceTransformer(path)
        local_val_preds, _ = get_model_predictions(model, "val")
        local_test_preds, _ = get_model_predictions(model, "test")
        val_predictions.append(local_val_preds)
        test_predictions.append(local_test_preds)

    val_predictions = np.array(val_predictions)
    test_predictions = np.array(test_predictions)

    val_predictions = np.mean(val_predictions, axis=0)
    test_predictions = np.mean(test_predictions, axis=0)

    best_validation_treshold = compute_f1_threhsold_for_label1(
        val_true_labels, val_predictions
    )

    # Compute the predictions of the ensemble with the given validation threshold

    val_predictions = [
        1 if x >= best_validation_treshold else 0 for x in val_predictions
    ]
    test_predictions = [
        1 if x >= best_validation_treshold else 0 for x in test_predictions
    ]

    print("Validation")
    _, _, f1, _ = precision_recall_fscore_support(val_true_labels, val_predictions)
    f1_score_class_1 = f1[1]

    # Calculate Accuracy
    accuracy = accuracy_score(val_true_labels, val_predictions)

    # Calculate AUC Score
    # Ensure that test_predictions contains probabilities for positive class
    auc_score = roc_auc_score(val_true_labels, val_predictions)
    print("Accuracy:", accuracy)
    print("AUC Score:", auc_score)
    print("F1 Score (Positive Labels):", f1_score_class_1)

    # Compute f1 score for class 1 on the test set

    _, _, f1, _ = precision_recall_fscore_support(test_true_labels, test_predictions)
    f1_score_class_1 = f1[1]
    # Calculate Accuracy
    accuracy = accuracy_score(test_true_labels, test_predictions)

    # Calculate AUC Score
    # Ensure that test_predictions contains probabilities for positive class
    auc_score = roc_auc_score(test_true_labels, test_predictions)
    print("Accuracy:", accuracy)
    print("AUC Score:", auc_score)
    print("F1 Score (Positive Labels):", f1_score_class_1)
    print("F1 Score (Positive Labels):", f1_score_class_1)


def hard_labels_ensemble(paths):
    val_predictions = []
    test_predictions = []

    _, _, val_true_labels = parse_groundtruth("val")
    _, _, test_true_labels = parse_groundtruth("test")

    for path in paths:
        model = SentenceTransformer(path)
        local_val_preds, val_threshold = get_model_predictions(model, "val")
        local_test_preds, _ = get_model_predictions(model, "test")
        val_predictions.append(local_val_preds)
        test_predictions.append(local_test_preds)

    val_predictions = np.array(val_predictions)
    test_predictions = np.array(test_predictions)

    val_predictions = mode(val_predictions, axis=0).mode
    test_predictions = mode(test_predictions, axis=0).mode

    best_validation_treshold = compute_f1_threhsold_for_label1(
        val_true_labels, val_predictions
    )

    val_predictions = [
        1 if x >= best_validation_treshold else 0 for x in val_predictions
    ]
    test_predictions = [
        1 if x >= best_validation_treshold else 0 for x in test_predictions
    ]

    # Compute the predictions of the ensemble with the given validation threshold

    # Compute f1 score for class 1 on the test set

    print("Validation")
    _, _, f1, _ = precision_recall_fscore_support(val_true_labels, val_predictions)
    f1_score_class_1 = f1[1]

    # Calculate Accuracy
    accuracy = accuracy_score(val_true_labels, val_predictions)

    # Calculate AUC Score
    # Ensure that test_predictions contains probabilities for positive class
    auc_score = roc_auc_score(val_true_labels, val_predictions)
    print("Accuracy:", accuracy)
    print("AUC Score:", auc_score)
    print("F1 Score (Positive Labels):", f1_score_class_1)

    print("Test")
    _, _, f1, _ = precision_recall_fscore_support(test_true_labels, test_predictions)
    f1_score_class_1 = f1[1]

    # Calculate Accuracy
    accuracy = accuracy_score(test_true_labels, test_predictions)

    # Calculate AUC Score
    # Ensure that test_predictions contains probabilities for positive class
    auc_score = roc_auc_score(test_true_labels, test_predictions)
    print("Accuracy:", accuracy)
    print("AUC Score:", auc_score)
    print("F1 Score (Positive Labels):", f1_score_class_1)


paths = [
    "./modelslr=1e-06_decay=0.01_opt=AdamW_sched=warmupcosinewithhardrestarts",
    "./modelslr=1e-06_decay=0_opt=AdamW_sched=warmupcosine",
    "./modelslr=1e-05_decay=0.01_opt=AdamW_sched=warmupcosine",
]

soft_labels_ensemble(paths)
hard_labels_ensemble(paths)
