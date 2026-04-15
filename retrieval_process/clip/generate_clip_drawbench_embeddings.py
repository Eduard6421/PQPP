# Script that extracts the positions of the captions that are similar to the best caption of each item


import pandas as pd
import numpy as np
import pickle
import torch
import os
import sys

from sklearn.metrics.pairwise import cosine_similarity
from pandas import json_normalize
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from tqdm import tqdm
# Dynamically add the parent directory of `retrieval_process` to `sys.path`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Now import `longclip` from its location within `retrieval_process`
from retrieval_process.clip.LongClip.model import longclip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = longclip.load("./LongClip/checkpoints/longclip-B.pt", device=device)


IMAGE_FOLDER = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\drawbench\drawbench_retrieval"
EMBEDDINGS_FOLDER = "./drawbench_embeddings"

def generate_image_embeddings(
    preprocess, IMAGE_FOLDER, EMBEDDINGS_FOLDER, batch_size=128
):
    print("Generating image embeddings...")
    prompt_numbers = [i for i in range(10000,10200)]
    batch_images = []
    batch_image_ids = []
    batch_counter = 0

    for idx, prompt_id in tqdm(enumerate(prompt_numbers)):

        # List  files from the folder {IMAGE_FOLDER}/prompt_id
        img_folder_path = f"{IMAGE_FOLDER}/{prompt_id}"

        for img_path in os.listdir(img_folder_path):
            # get the last part of img_path and cooncate with prompt_id like : {prompt_id}_{last_part}
            image_id = img_path.split('.')[0]
            img = Image.open(f"{img_folder_path}/{img_path}")
            img = img.convert("RGB")  # Ensure RGB format
            batch_images.append(preprocess(img).unsqueeze(0))  # Preprocess and add batch dim
            batch_image_ids.append(image_id)


            # Process the batch when it reaches the batch size
            if len(batch_images) == batch_size:
                print(f"Processing batch {batch_counter}")
                # Stack the batch along the first dimension
                batch_tensor = torch.cat(batch_images, dim=0).to(device)


                with torch.no_grad():
                    image_embeddings = model.encode_image(batch_tensor).cpu()

                print("Image embeddings shape:", image_embeddings.shape)

                # Save the batch embeddings
                batch_data = {"embeddings": image_embeddings, "image_ids": batch_image_ids}
                with open(
                    f"{EMBEDDINGS_FOLDER}/embeddings_batch_{batch_counter}.pkl", "wb"
                ) as file:
                    pickle.dump(batch_data, file)
                
                print(f"Wrote batch {batch_counter} to file")
                batch_counter += 1
                batch_images = []  # Reset the batch
                batch_image_ids = []

    # Process any remaining images
    if len(batch_images) > 0:
        batch_tensor = torch.cat(batch_images, dim=0).to(device)
        with torch.no_grad():
            image_embeddings = model.encode_image(batch_tensor).cpu()
        
        batch_data = {"embeddings": image_embeddings, "image_ids": batch_image_ids}

        print("Image embeddings shape:", image_embeddings.shape)

        with open(
            f"{EMBEDDINGS_FOLDER}/embeddings_batch_{batch_counter}.pkl", "wb"
        ) as file:
            pickle.dump(batch_data, file)
        print(f"Wrote final batch {batch_counter} to file")


generate_image_embeddings(
    preprocess=processor,
    IMAGE_FOLDER=IMAGE_FOLDER,
    EMBEDDINGS_FOLDER=EMBEDDINGS_FOLDER,
    batch_size=1024,
)
