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

dataset_path = "../dataset/sentence_transformer_dataset.pickle"


def finetune_sentence_transformer():
    # Load train/val/test examples
    with open("../dataset/train_examples_new.pickle", "rb") as f:
        train_examples = pickle.load(f)
    with open("../dataset/val_examples_new.pickle", "rb") as f:
        val_examples = pickle.load(f)

    # Create dataloaders
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=256)

    val_sentences_1 = []
    val_sentences_2 = []
    val_scores = []
    for item in val_examples:
        sentence_1 = item.texts[0]
        sentence_2 = item.texts[1]
        label = item.label
        val_sentences_1.append(sentence_1)
        val_sentences_2.append(sentence_2)
        val_scores.append(label)

    # Load or define model
    model = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device="cuda"
    )

    # Define the training loss
    train_loss = losses.CosineSimilarityLoss(model)

    # Prepare for evaluator
    evaluator = evaluation.BinaryClassificationEvaluator(
        val_sentences_1, val_sentences_2, val_scores
    )

    # Grid search parameters
    learning_rates = [1e-4]
    weight_decays = [
        0.01,
        0.1,
        0,
    ]
    optimizers = [torch.optim.AdamW]
    schedulers = ["warmupcosine", "warmupcosinewithhardrestarts"]

    # Iterate over all combinations
    for lr, decay, opt_class, scheduler in itertools.product(
        learning_rates, weight_decays, optimizers, schedulers
    ):
        # Load model
        model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2", device="cuda"
        )

        # Define the training loss
        train_loss = losses.CosineSimilarityLoss(model)

        # Prepare for evaluator
        evaluator = evaluation.BinaryClassificationEvaluator(
            val_sentences_1, val_sentences_2, val_scores
        )

        # Model save path
        model_save_path = (
            f"./models/lr={lr}_decay={decay}_opt={opt_class.__name__}_sched={scheduler}"
        )
        os.makedirs(model_save_path, exist_ok=True)

        # Model training
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=12,
            optimizer_class=opt_class,
            optimizer_params={"lr": lr, "weight_decay": decay},
            scheduler=scheduler,
            warmup_steps=10,
            output_path=model_save_path,
            save_best_model=True,
            evaluation_steps=50,
            show_progress_bar=True,
        )


# finetune_sentence_transformer()
