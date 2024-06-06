import itertools
import pickle
import random
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers import evaluation, util
from sklearn.metrics import roc_curve, auc
from collections import Counter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve


dataset_path = "../dataset/sentence_transformer_dataset_new.pickle"


def cosine_similarity(embeddings1, embeddings2):
    return np.dot(embeddings1, embeddings2) / (
        np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2)
    )


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


def evaluate_sentence_transformer(
    path, evaluation_type="val", model_name=None, best_threshold=None
):
    # eval_csv = pd.read_csv(path + "/eval/binary_classification_evaluation_results.csv")

    # Calculate the maximum score for each row among the specified columns
    # eval_csv["max_score"] = eval_csv[score_column_names].max(axis=1)

    # Find the row with the highest of these maximum scores
    # row_with_highest_score = eval_csv.loc[eval_csv["max_score"].idxmax()]
    # threshold = row_with_highest_score["cossim_accuracy_threshold"]

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
        best_threshold = compute_f1_threhsold_for_label1(true_labels, predictions)

    print("Threshold value")
    print(best_threshold)

    # Compute ROC curve and ROC area
    fpr, tpr, roc_auc = roc_curve(true_labels, predictions)
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

    fig = plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.title(f"Receiver Operating Characteristic - {model_name}", fontsize=20)
    plt.legend(loc="lower right", fontsize=20)

    # Save the figure as a PDF file
    # plt.savefig("ROC_Curve_Base.pdf", bbox_inches="tight")

    print("Evaluation Suite:", evaluation_type)
    print("Accuracy:", correct / len(true_labels))
    print("AUC Score:", roc_auc)
    print("F1 Score (Positive Labels):", f1_score_class_1)

    return (best_threshold, fpr, tpr, roc_auc, fig)


def evaluation_suite():
    paths = (
        "sentence-transformers/all-mpnet-base-v2"
        # "./models/lr=0.0001_decay=0_opt=AdamW_sched=warmupcosine",
        # "./models/lr=0.0001_decay=0_opt=AdamW_sched=warmupcosinewithhardrestarts",
        # "./models/lr=0.0001_decay=0.01_opt=AdamW_sched=warmupcosine",
        # "./models/lr=0.0001_decay=0.01_opt=AdamW_sched=warmupcosinewithhardrestarts",
        # "./models/lr=1e-05_decay=0_opt=AdamW_sched=warmupcosine",
        # "./models/lr=1e-05_decay=0_opt=AdamW_sched=warmupcosinewithhardrestarts",
        # "./models/lr=1e-05_decay=0.01_opt=AdamW_sched=warmupcosine",
        # "./models/lr=1e-05_decay=0.01_opt=AdamW_sched=warmupcosinewithhardrestarts",
        # "./models/lr=1e-06_decay=0_opt=AdamW_sched=warmupcosine",
        # "./models/lr=1e-06_decay=0_opt=AdamW_sched=warmupcosinewithhardrestarts",
        # "./models/lr=1e-06_decay=0.01_opt=AdamW_sched=warmupcosine",
        "./models/lr=1e-06_decay=0.01_opt=AdamW_sched=warmupcosinewithhardrestarts",
    )

    thd, fpr1, tpr1, roc_auc1, draw1 = evaluate_sentence_transformer(
        path="sentence-transformers/all-mpnet-base-v2", model_name="all-mpnet-base-v2"
    )
    _, fpr2, tpr2, roc_auc2, draw2 = evaluate_sentence_transformer(
        path="./modelslr=1e-06_decay=0.01_opt=AdamW_sched=warmupcosinewithhardrestarts",
        model_name="Finetuned sentence transformer",
    )

    print(thd)

    model_names = ["Pre-trained sentence BERT", "Fine-tuned sentence BERT"]

    with PdfPages("Combined_ROC_Curves.pdf") as pdf:
        plt.figure(figsize=(10, 6))
        # Plot ROC curve for each model
        for fpr, tpr, model_name, auc_score in zip(
            [fpr1, fpr2], [tpr1, tpr2], model_names, [roc_auc1, roc_auc2]
        ):
            plt.plot(
                fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc_score:.2f})"
            )

        plt.plot([0, 1], [0, 1], "k--", linewidth=2)  # Dashed diagonal for reference
        # Set plot title and axis labels
        plt.title("ROC Curve Comparison", fontsize=20)
        plt.xlabel("False Positive Rate", fontsize=20)
        plt.ylabel("True Positive Rate", fontsize=20)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        # ax.tick_params(axis='both', which='major', labelsize=10)
        # ax.tick_params(axis='both', which='minor', labelsize=8)
        # change xtick size
        # plt.xticks(fontsize=20)
        # plt.yticks(font_)
        plt.legend(loc="lower right", fontsize=18)

        # Save the combined ROC curve plot to a PDF file
        pdf.savefig(bbox_inches="tight")
    plt.close()


# print("Model type: Pretrained")
# best_threshold = evaluate_sentence_transformer(
#    "sentence-transformers/all-mpnet-base-v2", "val"
# )
# evaluate_sentence_transformer(
#    "sentence-transformers/all-mpnet-base-v2", "test", best_threshold=best_threshold
# )
evaluation_suite()
