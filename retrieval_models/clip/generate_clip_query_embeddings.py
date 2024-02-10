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

    return text_embeddings


def run_clip_retrieval(num_queries, model, processor, dataframe):
    result_array = []
    for i in range(num_queries):
        print(f"Generating retrieval results for item, {i}")
        row = dataframe.iloc[i]
        caption = row["best_caption"]
        embeddings = retrieve_images(model=model, processor=processor, text=caption)
        result_array.append(embeddings)

    return result_array


retrieval_results = run_clip_retrieval(10000, model, processor, best_captions_df)
pickle.dump(retrieval_results, open(EMBEDDINGS_FOLDER + "embeddings.pickle", "wb"))
