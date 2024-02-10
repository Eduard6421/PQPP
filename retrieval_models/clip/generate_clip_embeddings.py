# Script that extracts the positions of the captions that are similar to the best caption of each item


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pandas import json_normalize
from PIL import Image
import pickle

from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
import torch

# Dataframe containing each annotation
df_annotations_path = "../../dataset/df_annotations.pkl"
df_annotations = pd.read_pickle(df_annotations_path)

# dataframe that contains the best caption for each item
best_caption_path = "../../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path)

# A set of best_caption_id and image_id matches
query_matches_path = "../../dataset/query_matches.pickle"
query_matches = pd.read_pickle(query_matches_path)

IMAGE_FOLDER = "../../dataset/train2017/train2017"
EMBEDDINGS_FOLDER = "./clip_image_embeddings"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


model = model.to("cuda")


# Generate the image embeddings and save them to disk
def generate_image_embeddings(
    image_processor, IMAGE_FOLDER, EMBEDDINGS_FOLDER, batch_size=128
):
    print("Generating image embeddings...")
    image_ids = df_annotations.image_id.unique()
    batch_images = []
    batch_image_ids = []
    batch_counter = 0

    for idx, image_id in enumerate(image_ids):
        if idx % 100 == 0:
            print(idx)
        temp_id = str(image_id).zfill(12)
        img = Image.open(f"{IMAGE_FOLDER}/{temp_id}.jpg")
        batch_images.append(img)
        batch_image_ids.append(image_id)

        # Process the batch when it reaches the batch size
        if len(batch_images) == batch_size:
            print(f"Processing batch {batch_counter}")
            inputs = image_processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
            pixel_values = pixel_values.to("cuda")
            embeddings = (
                model.get_image_features(pixel_values=pixel_values, return_dict=True)
                .detach()
                .cpu()
            )

            batch_data = {"embeddings": embeddings, "image_ids": batch_image_ids}
            with open(
                f"{EMBEDDINGS_FOLDER}/embeddings_batch_{batch_counter}.pkl", "wb"
            ) as file:
                pickle.dump(batch_data, file)
            print(f"Wrote batch {batch_counter} to file")
            batch_counter += 1
            batch_images = []  # Reset the batch
            batch_image_ids = []

    # Process the remaining images if any
    if len(batch_images) > 0:
        inputs = image_processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        pixel_values = pixel_values.to("cuda")
        embeddings = (
            model.get_image_features(pixel_values=pixel_values, return_dict=True)
            .detach()
            .cpu()
        )

        batch_data = {"embeddings": embeddings, "image_ids": batch_image_ids}
        with open(
            f"{EMBEDDINGS_FOLDER}/embeddings_batch_{batch_counter}.pkl", "wb"
        ) as file:
            pickle.dump(batch_data, file)


generate_image_embeddings(
    image_processor=image_processor,
    IMAGE_FOLDER=IMAGE_FOLDER,
    EMBEDDINGS_FOLDER=EMBEDDINGS_FOLDER,
    batch_size=512,
)
