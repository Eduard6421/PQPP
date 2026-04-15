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


train_queries_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_train.csv"
val_queries_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_val.csv"
test_queries_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_test.csv"

train_df = pd.read_csv(train_queries_path)
val_df = pd.read_csv(val_queries_path)
test_df = pd.read_csv(test_queries_path)


model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor",
    model_type="pretrain_vitL",
    is_eval=True,
    device=device,
)

def generate_query_embedding(model, processsor, text_batch):
    text_processed_batch = []
    for text in text_batch:
        text_input = txt_processors["eval"](text)
        text_processed_batch.append(text_input)
    sample = {"text_input": text_processed_batch}

    with torch.no_grad():
        text_embeddings = model.extract_features(sample, mode="text")
    cls_embedding = text_embeddings.text_embeds[:, 0, :]

    return cls_embedding


def generate_blip2_query_embeddings(model, processor, dataframe):

    best_captions = dataframe["caption"].tolist()

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


train_embeddings = generate_blip2_query_embeddings( model, vis_processors, train_df)
val_embeddings = generate_blip2_query_embeddings( model, vis_processors, val_df)
test_embeddings = generate_blip2_query_embeddings( model, vis_processors, test_df)


pickle.dump(train_embeddings, open("./blip2_query_embeddings/train_embeddings.pickle", "wb"))
pickle.dump(val_embeddings, open("./blip2_query_embeddings/val_embeddings.pickle", "wb"))
pickle.dump(test_embeddings, open("./blip2_query_embeddings/test_embeddings.pickle", "wb"))
