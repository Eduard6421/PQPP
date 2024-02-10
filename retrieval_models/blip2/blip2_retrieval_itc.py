from multiprocessing import Pool
import torch
import pickle
import pandas as pd
import numpy as np

from lavis.models import load_model_and_preprocess
from PIL import Image
from tqdm import tqdm

IMAGE_FOLDER = "../../dataset/train2017/train2017"
EMBEDDINGS_FOLDER = "./blip2_image_embeddings"


# from lavis.models import model_zoo
# dataframe that contains the best caption for each item


# Dataframe containing each annotation from MS-COCO


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


def split_into_batches(image_ids, batch_size):
    """Splits a list of image IDs into smaller batches.

    Args:
        image_ids (list): The list of image IDs.
        batch_size (int): The size of each batch.

    Returns:
        list of list: A list containing smaller batches of image IDs.
    """
    return [image_ids[i : i + batch_size] for i in range(0, len(image_ids), batch_size)]


def split_into_batches(image_ids, batch_size):
    """Splits a list of image IDs into smaller batches.
    Args:
        image_ids (list): The list of image IDs.
        batch_size (int): The size of each batch.
    Returns:
        list of list: A list containing smaller batches of image IDs.
    """
    return [image_ids[i : i + batch_size] for i in range(0, len(image_ids), batch_size)]


def load_image(image_path):
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")  # or any other processing
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_images_in_parallel(image_paths, image_processor, num_processes):
    with Pool(num_processes) as p:
        image_list = p.map(load_image, image_paths, num_processes)
    # Convert PIL images to tensors
    image_tensors = torch.stack([image_processor(img) for img in image_list])
    # Reshape the tensor to lenXimageshape
    return image_tensors


num_processes = 11  # Adjust based on your CPU


def retrieve_images(
    model,
    vis_processor,
    text_processor,
    text,
    image_ids,
    device,
    batch_size=32,
):
    # Process text
    txt = text_processor(text)
    all_scores = []
    all_batch_image_ids = []
    image_batches = split_into_batches(image_ids, batch_size=batch_size)

    # Process each batch
    for image_batch in tqdm(image_batches):
        image_paths = [
            f"{IMAGE_FOLDER}/{str(image_id).zfill(12)}.jpg" for image_id in image_batch
        ]
        images_batch_tensor = load_images_in_parallel(
            image_paths, vis_processor, num_processes
        ).to(device)

        with torch.no_grad():
            scores = model(
                {
                    "image": images_batch_tensor,
                    "text_input": [txt for _ in range(len(image_batch))],
                },
                match_head="itc",
            ).squeeze()

        all_scores.extend(scores.tolist())
        all_batch_image_ids.extend(image_batch)

    # Sort the results
    sorted_scores_indices = np.argsort(all_scores)[::-1]
    sorted_image_ids = [all_batch_image_ids[i] for i in sorted_scores_indices]
    sorted_scores = [all_scores[i] for i in sorted_scores_indices]

    return sorted_image_ids, sorted_scores


def run_blip2_retrieval(
    num_queries, model, vis_processor, text_processor, dataframe, image_ids
):
    result_map = {}
    for i in tqdm(range(num_queries)):
        row = dataframe.iloc[i]
        caption = row["best_caption"]
        sorted_image_ids, sorted_scores = retrieve_images(
            model=model,
            vis_processor=vis_processor,
            text_processor=text_processor,
            text=caption,
            image_ids=image_ids,
            batch_size=32,
            device=device,
        )
        result_map[i] = sorted_image_ids

    return result_map


if __name__ == "__main__":
    df_annotations_path = "../../dataset/df_annotations.pkl"
    best_caption_path = "../../dataset/best_captions_df.pickle"

    df_annotations = pd.read_pickle(df_annotations_path)
    best_captions_df = pd.read_pickle(best_caption_path)
    image_ids = df_annotations.image_id.unique()

    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip2_image_text_matching", "pretrain", device=device, is_eval=True
    )

    retrieval_results = run_blip2_retrieval(
        num_queries=10000,
        model=model,
        vis_processor=vis_processors["eval"],
        text_processor=text_processors["eval"],
        dataframe=best_captions_df,
        image_ids=image_ids,
    )
    pickle.dump(retrieval_results, open("blip2_retrieval_results_itc.pickle", "wb"))
