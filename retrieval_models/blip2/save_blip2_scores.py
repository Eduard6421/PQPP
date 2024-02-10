import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from pandas import json_normalize
from concurrent.futures import ThreadPoolExecutor, as_completed

tensor = torch.Tensor
device = torch.device("cpu")

# dataframe that contains the best caption for each item
best_caption_path = "../../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path).head(10000)

IMAGE_FOLDER = "../../dataset/train2017/train2017"
EMBEDDINGS_FOLDER = "./blip2_image_embeddings"
QUERY_EMBEDDINGS = "./blip2_query_embeddings/embeddings.pickle"


query_embeddings_df = pd.read_pickle(QUERY_EMBEDDINGS)


def retrieve_images(current_text_embedding):
    # Similarity score with each image
    similarity_scores = []
    image_ids = []
    for i in range(0, 116):
        image_embedding_file = f"{EMBEDDINGS_FOLDER}/embeddings_batch_{i}.pkl"
        stored_embeddings = pickle.load(open(image_embedding_file, "rb"))
        image_embeddings = stored_embeddings["embeddings"]

        # similarities = cosine_similarity(image_embeddings["embeddings"], cls_embedding)
        # image embeddings: BATCH_SIZE x QUERY_NUM x 768
        # text embeddings: 1 x NUM_TOKEN x 768

        # create a batch_size x 768 tensor with -inf in all positions

        similarities = np.full((image_embeddings.shape[0], 1), -np.inf)

        for query_token in range(image_embeddings.shape[1]):
            # print(image_embeddings[:, query_token, :].shape)
            # print(cls_embedding.shape)

            query_similarities = cosine_similarity(
                image_embeddings[:, query_token, :], [current_text_embedding]
            )
            mask = query_similarities > similarities
            similarities[mask] = query_similarities[mask]

        # squash second dimension
        similarities = similarities.squeeze(1)

        for idx, score in enumerate(similarities):
            similarity_scores.append(score)
            image_ids.append(stored_embeddings["image_ids"][idx])
    sorted_indices = np.argsort(similarity_scores)[::-1]  # Sort in descending order
    sorted_scores = np.array(similarity_scores)[sorted_indices]
    sorted_image_ids = np.array([image_ids[idx] for idx in sorted_indices])

    return sorted_scores


def save_blip2_scores_parallel(query_embeddings_df, num_queries, max_workers=12):
    result_array = [None] * num_queries  # Preallocate list for results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        future_to_index = {
            executor.submit(retrieve_images, query_embeddings_df[i]): i
            for i in range(0, num_queries)
        }

        # Retrieve results as tasks complete
        for future in tqdm(
            as_completed(future_to_index),
            total=num_queries,
            desc="Generating retrieval scores",
        ):
            i = future_to_index[future]
            try:
                result_array[
                    i
                ] = future.result()  # Store the result based on the original index
            except Exception as e:
                print(f"Error processing index {i}: {e}")
    return result_array


if __name__ == "__main__":
    retrieval_results = save_blip2_scores_parallel(query_embeddings_df, 10000)
    best_captions_df = best_captions_df.head(10000)
    best_captions_df["blip2_scores"] = retrieval_results
    best_captions_df.to_pickle("./blip2_scores_df.pickle")
