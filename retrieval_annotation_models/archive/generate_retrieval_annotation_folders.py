# Script that extracts the positions of the captions that are similar to the best caption of each item

from collections import Counter
import pandas as pd
import numpy as np
import shutil
import os
import re
import spacy

from sklearn.metrics.pairwise import cosine_similarity
from pandas import json_normalize
from joblib import dump
from tqdm import tqdm

df_annotations_path = "../dataset/df_annotations.pkl"
df_annotations = pd.read_pickle(df_annotations_path)

best_caption_path = "../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path)

query_matches_path = "../dataset/query_matches.pickle"

nlp = spacy.load("en_core_web_lg")

RETRIEVAL_MATCHES_UPPER_LIMIT = 200


def generate_bag_of_words(sentence):
    # Process the sentence using spaCy
    doc = nlp(sentence)

    # Lemmatize and filter out punctuation and stop words
    lemmatized_words = [
        token.lemma_ for token in doc if not token.is_punct and not token.is_stop
    ]

    # Generate the bag of words
    bag_of_words = Counter(lemmatized_words)

    return bag_of_words


def all_words_are_present(query_bow, retrieval_bow):
    query_bow_set = set(query_bow.keys())
    retrieval_bow_set = set(retrieval_bow.keys())

    return query_bow_set.issubset(retrieval_bow_set)


def recompute_matches(caption_text, found_ids):
    # print(caption_text)

    true_matches = set()
    query_bow = generate_bag_of_words(caption_text)
    # Iterate through found_ids
    for image_ids in found_ids:
        # Get the captions for a certain image_id
        items = df_annotations[df_annotations["image_id"] == image_ids]["caption"]

        retrieved_bow = Counter()
        for caption in items:
            print(caption)
            retrieved_bow.update(generate_bag_of_words(caption))

        if all_words_are_present(query_bow, retrieved_bow):
            true_matches.add(image_ids)

    return true_matches


def check_match(caption_text, id):
    # Get the captions for a certain image_id
    items = df_annotations[df_annotations["image_id"] == id]["caption"]

    # print(caption_text)

    retrieved_bow = Counter()
    for caption in items:
        # print(caption)
        retrieved_bow.update(generate_bag_of_words(caption))

    query_bow = generate_bag_of_words(caption_text)

    # print(query_bow)
    # print(retrieved_bow)
    # print(all_words_are_present(query_bow, retrieved_bow))

    return all_words_are_present(query_bow, retrieved_bow)


def generate_retrieval_sets(
    best_captions_df, all_captions_df, output_path, num_items, min_similarity
):
    # Only the first 10k items
    best_captions = best_captions_df.head(num_items)["caption_embeddings"]
    all_captions = all_captions_df["caption_embeddings"]

    converted_best_captions = np.array([np.array(item) for item in best_captions])
    converted_all_captions = np.array([np.array(item) for item in all_captions])

    del best_captions
    del all_captions

    sim = cosine_similarity(converted_best_captions, converted_all_captions)

    for query_index in tqdm(range(900, 1000)):
        row = best_captions_df.iloc[query_index]
        sim_array = sim[query_index]

        # Get the index of the descending sorted array
        sorted_sim_array = np.argsort(sim_array)[::-1]

        # find index of the first item that is below the min_similarity
        cutoff_index = np.where(sim_array[sorted_sim_array] < min_similarity)[0][0]

        # Replace or remove invalid characters
        caption_text = row["best_caption"]
        invalid_chars = r'.<>:"/\|?*'
        for ch in invalid_chars:
            caption_text = caption_text.replace(ch, "")

        # create the directory  dest_folder_path
        dest_folder_path = os.path.join(
            "../retrieval_query_groundtruth",
            f"{query_index}_{caption_text}".strip(),
        )

        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)

        selected_subset = sorted_sim_array[:cutoff_index]

        sorted_scores = sim_array[selected_subset]

        found_ids = set()

        for retrieval_index, all_captions_index in enumerate(selected_subset):
            annot = all_captions_df.iloc[all_captions_index]
            image_id = annot["image_id"]

            if image_id in found_ids:
                continue

            if len(found_ids) >= RETRIEVAL_MATCHES_UPPER_LIMIT:
                if check_match(caption_text, image_id) == False:
                    continue

            # Copy over the images in order of the scores.
            src_id = str(image_id).zfill(12)
            src_path = os.path.join("../dataset/train2017/train2017/", f"{src_id}.jpg")

            dest_id = f"{retrieval_index}_{sorted_scores[retrieval_index]}_{src_id}"
            dest_path = os.path.join(dest_folder_path, f"{dest_id}.jpg")
            # print("writing image", src_path, "to", dest_path)
            shutil.copy(src_path, dest_path)

            found_ids.add(image_id)


generate_retrieval_sets(
    best_captions_df, df_annotations, query_matches_path, 10000, 0.70
)
