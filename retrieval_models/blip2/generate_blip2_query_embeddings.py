import pandas as pd
import numpy as np
import pickle
import torch

from sklearn.metrics.pairwise import cosine_similarity
from pandas import json_normalize
from PIL import Image
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

tensor = torch.Tensor
device = torch.device("cpu")

# dataframe that contains the best caption for each item
best_caption_path = "../../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path)

IMAGE_FOLDER = "../../dataset/train2017/train2017"
QUERY_EMBEDDINGS_FOLDER = "./blip2_query_embeddings"

model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor",
    model_type="pretrain_vitL",
    is_eval=True,
    device=device,
)


def generate_query_embedding(model, processsor, text_batch):
    # print(text)

    text_processed_batch = []
    for text in text_batch:
        text_input = txt_processors["eval"](text)
        text_processed_batch.append(text_input)
    sample = {"text_input": text_processed_batch}

    with torch.no_grad():
        text_embeddings = model.extract_features(sample, mode="text")
    cls_embedding = text_embeddings.text_embeds[:, 0, :]

    return cls_embedding


def generate_blip2_query_embeddings(num_queries, model, processor, dataframe):
    dataframe = dataframe.head(10000)

    best_captions = dataframe["best_caption"].tolist()

    # Split into batches
    batch_size = 128
    batches = []

    for i in range(0, len(best_captions), batch_size):
        batches.append(best_captions[i : i + batch_size])

    embeddings = []
    for batch in tqdm(batches):
        embeddings.append(generate_query_embedding(model, processor, batch))

    embeddings = torch.cat(embeddings, dim=0)

    return embeddings


embeddings = generate_blip2_query_embeddings(
    10000, model, vis_processors, best_captions_df
)

pickle.dump(embeddings, open("./blip2_query_embeddings/embeddings.pickle", "wb"))
