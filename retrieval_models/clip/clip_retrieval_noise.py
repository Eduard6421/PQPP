# Script that extracts the positions of the captions that are similar to the best caption of each item


import pandas as pd
import numpy as np
import pickle
import torch

from sklearn.metrics.pairwise import cosine_similarity
from pandas import json_normalize
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor


# dataframe that contains the best caption for each item
best_caption_path = "../../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path)

IMAGE_FOLDER = "../../dataset/train2017/train2017"
EMBEDDINGS_FOLDER = "./clip_image_embeddings"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def retrieve_images(model, processor, text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    with torch.no_grad():
        text_embeddings = model.get_text_features(**inputs)[0]

    # Similarity score with each image
    similarity_scores = []
    # The Image ids
    image_ids = []
    # A bit of refactor works here.
    # You can replace the 232 by the num of items / batch_size
    for i in range(0, 232):
        image_embedding_file = f"{EMBEDDINGS_FOLDER}/embeddings_batch_{i}.pkl"
        image_embeddings = pickle.load(open(image_embedding_file, "rb"))

        # add noise to text_embeddings
        noise = torch.randn(text_embeddings.shape)
        text_embeddings = text_embeddings + noise * 0.1

        similarities = cosine_similarity(
            np.expand_dims(text_embeddings, axis=0), image_embeddings["embeddings"]
        ).flatten()

        for idx, score in enumerate(similarities):
            similarity_scores.append(score)
            image_ids.append(image_embeddings["image_ids"][idx])

    # Sorting
    sorted_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
    sorted_scores = np.array(similarity_scores)[sorted_indices]

    # Retrieve original global indices
    sorted_image_ids = np.array([image_ids[idx] for idx in sorted_indices])

    return sorted_image_ids, sorted_scores


def run_clip_retrieval(num_queries, model, processor, dataframe):
    result_map = {}
    for i in range(8000, 10000):
        print(f"Generating retrieval results for item, {i}")
        row = dataframe.iloc[i]
        caption = row["best_caption"]
        sorted_image_ids, sorted_scores = retrieve_images(
            model=model, processor=processor, text=caption
        )
        result_map[i] = sorted_image_ids

    return result_map


retrieval_results = run_clip_retrieval(10000, model, processor, best_captions_df)
pickle.dump(retrieval_results, open("./clip_retrieval_results_noised.pickle", "wb"))
