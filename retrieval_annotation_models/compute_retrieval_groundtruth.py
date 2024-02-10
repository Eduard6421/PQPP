# Script that extracts the positions of the captions that are similar to the best caption of each item

from collections import Counter
import pandas as pd
import numpy as np
import shutil
import os
import re
import spacy
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from pandas import json_normalize
from joblib import dump
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses

df_annotations_path = "../dataset/df_annotations.pkl"
df_annotations = pd.read_pickle(df_annotations_path)

best_caption_path = "../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path)

query_matches_path = "../dataset/query_matches.pickle"

THRESHOLD_VALUE = 0.5206602


def generate_retrieval_sets(best_captions_df, all_captions_df, output_path, num_items):
    model = SentenceTransformer(
        "./modelslr=1e-06_decay=0.01_opt=AdamW_sched=warmupcosinewithhardrestarts"
    )
    ground_truth = {}

    # Only the first 10k items
    best_captions = best_captions_df.head(num_items)["best_caption"]
    all_captions = all_captions_df["caption"]

    best_captions_encoded = [
        model.encode(caption)
        for caption in tqdm(best_captions, desc="Encoding best captions")
    ]
    all_captions_encoded = [
        model.encode(caption)
        for caption in tqdm(all_captions, desc="Encoding all captions")
    ]

    ground_truth = {}

    total_positive = 0
    total_negative = 0

    for i in tqdm(
        range(1000, len(best_captions_encoded)), desc="Computing ground truths"
    ):
        encoded_caption = best_captions_encoded[i]
        similarities = cosine_similarity([encoded_caption], all_captions_encoded)[0]
        matches = np.where(similarities >= THRESHOLD_VALUE)[0]
        negative_matches = np.where(similarities < THRESHOLD_VALUE)[0]

        total_positive += len(matches)
        total_negative += len(negative_matches)

        # Select the image_id of the matches
        matches = all_captions_df.iloc[matches]["image_id"].values
        ground_truth[i] = list(set(matches))

    print("Total positive matches:", total_positive)
    print("Total negative matches:", total_negative)
    print("Total data annotated:", total_positive + total_negative)

    with open(output_path, "wb") as handle:
        pickle.dump(ground_truth, handle, protocol=pickle.HIGHEST_PROTOCOL)


generate_retrieval_sets(best_captions_df, df_annotations, query_matches_path, 10000)
