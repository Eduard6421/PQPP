import pandas as pd
import numpy as np
import pickle
import torch

from sklearn.metrics.pairwise import cosine_similarity
from pandas import json_normalize
from PIL import Image
from lavis.models import load_model_and_preprocess

tensor = torch.Tensor
device = torch.device("cpu")

# dataframe that contains the best caption for each item
best_caption_path = "../../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path)

IMAGE_FOLDER = "../../dataset/train2017/train2017"
EMBEDDINGS_FOLDER = "./blip2_image_embeddings"

model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor",
    model_type="pretrain_vitL",
    is_eval=True,
    device=device,
)


def retrieve_images(model, processor, text):
    # print(text)
    text_input = txt_processors["eval"](text)
    sample = {"text_input": [text_input]}

    with torch.no_grad():
        text_embeddings = model.extract_features(sample, mode="text")

    # Similarity score with each image
    similarity_scores = []
    image_ids = []
    for i in range(0, 116):
        image_embedding_file = f"{EMBEDDINGS_FOLDER}/embeddings_batch_{i}.pkl"
        stored_embeddings = pickle.load(open(image_embedding_file, "rb"))
        image_embeddings = stored_embeddings["embeddings"]

        # Compute cosine similarity
        cls_embedding = text_embeddings.text_embeds[:, 0, :]

        # add noise to cls_embedding
        cls_embedding = cls_embedding + torch.randn(cls_embedding.shape)

        # similarities = cosine_similarity(image_embeddings["embeddings"], cls_embedding)
        # image embeddings: BATCH_SIZE x QUERY_NUM x 768
        # text embeddings: 1 x NUM_TOKEN x 768

        # create a batch_size x 768 tensor with -inf in all positions

        similarities = np.full((image_embeddings.shape[0], 1), -np.inf)

        for query_token in range(image_embeddings.shape[1]):
            # print(image_embeddings[:, query_token, :].shape)
            # print(cls_embedding.shape)

            query_similarities = cosine_similarity(
                image_embeddings[:, query_token, :], cls_embedding
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

    return sorted_image_ids, sorted_scores


def run_blip2_retrieval(num_queries, model, processor, dataframe):
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


retrieval_results = run_blip2_retrieval(10000, model, vis_processors, best_captions_df)
pickle.dump(retrieval_results, open("blip2_retrieval_results_noised.pickle", "wb"))
