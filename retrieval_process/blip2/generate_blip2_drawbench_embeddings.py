import pandas as pd
import numpy as np
import pickle
import torch
import os 
from sklearn.metrics.pairwise import cosine_similarity
from pandas import json_normalize
from PIL import Image
from tqdm import tqdm

# pip install salesforce-lavis
from lavis.models import load_model_and_preprocess


IMAGE_FOLDER = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\drawbench\drawbench_retrieval"
EMBEDDINGS_FOLDER = "./drawbench_embeddings"

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor",
    model_type="pretrain_vitL",
    is_eval=True,
    device=device,
)

model = model.to("cuda")


def generate_image_embeddings(
    vis_processors, IMAGE_FOLDER, EMBEDDINGS_FOLDER, batch_size=128
):
    print("Generating image embeddings...")
    prompt_ids = [i for i in range(10000,10200)]
    batch_images = []
    batch_image_ids = []
    batch_counter = 0

    for idx, prompt_id in tqdm(enumerate(prompt_ids)):

        # List  files from the folder {IMAGE_FOLDER}/image_id

        img_folder_path = f"{IMAGE_FOLDER}/{prompt_id}"

        for img_path in os.listdir(img_folder_path):
            img = Image.open(f"{img_folder_path}/{img_path}")
            image_id  = img_path.split('.')[0]
            batch_images.append(img)
            batch_image_ids.append(image_id)


            # Process the batch when it reaches the batch size
            if len(batch_images) == batch_size:
                print(f"Processing batch {batch_counter}")

                image_embeddings = []
                for image in batch_images:
                    with torch.no_grad():
                        image = image.convert("RGB")
                        image_embedding = (
                            vis_processors["eval"](image).unsqueeze(0).to(device)
                        )
                        image_embeddings.append(image_embedding)

                # Use torch.stack to combine tensors into a single tensor
                image_embeddings_tensor = torch.stack(image_embeddings, dim=0)
                # REMOVE DIMENSION 1
                image_embeddings_tensor = image_embeddings_tensor.squeeze(1)

                sample = {"image": image_embeddings_tensor}
                # Using torch.no_grad() to disable gradient computation
                with torch.no_grad():
                    embeddings = (
                        model.extract_features(sample, mode="image")
                        .image_embeds.detach()
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
            image_embeddings = []
            for image in batch_images:
                image = image.convert("RGB")
                image_embedding = vis_processors["eval"](image).unsqueeze(0).to(device)
                image_embeddings.append(image_embedding)
            # Use torch.stack to combine tensors into a single tensor
            image_embeddings_tensor = torch.stack(image_embeddings, dim=0)

            # REMOVE DIMENSION 1
            image_embeddings_tensor = image_embeddings_tensor.squeeze(1)
            sample = {"image": image_embeddings_tensor}
            # Using torch.no_grad() to disable gradient computation
            with torch.no_grad():
                embeddings = (
                    model.extract_features(sample, mode="image").image_embeds.detach().cpu()
                )

            batch_data = {"embeddings": embeddings, "image_ids": batch_image_ids}

            with open(
                f"{EMBEDDINGS_FOLDER}/embeddings_batch_{batch_counter}.pkl", "wb"
            ) as file:
                pickle.dump(batch_data, file)


generate_image_embeddings(
    vis_processors=vis_processors,
    IMAGE_FOLDER=IMAGE_FOLDER,
    EMBEDDINGS_FOLDER=EMBEDDINGS_FOLDER,
    batch_size=1024,
)
