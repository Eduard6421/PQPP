

# Script that extracts the positions of the captions that are similar to the best caption of each item


import pandas as pd
import numpy as np
import pickle
import torch

from sklearn.metrics.pairwise import cosine_similarity
from pandas import json_normalize
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
from tqdm import tqdm

import sys
import os

# Dynamically add the parent directory of `retrieval_process` to `sys.path`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Now import `longclip` from its location within `retrieval_process`
from retrieval_process.clip.LongClip.model import longclip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = longclip.load("./LongClip/checkpoints/longclip-B.pt", device=device)

EMBEDDINGS_FOLDER = "./clip_query_embeddings/"

train_queries_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_train.csv"
val_queries_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_val.csv"
test_queries_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_test.csv"

train_df = pd.read_csv(train_queries_path)
val_df = pd.read_csv(val_queries_path)
test_df = pd.read_csv(test_queries_path)

def retrieve_embeddings(model, text):
    text = longclip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)[0]
    return text_features

def run_clip_retrieval( model, df):
    result_array = []
    for i in tqdm(range(df.shape[0])):
        row = df.iloc[i]
        caption = row["caption"]
        embeddings = retrieve_embeddings(model=model, text=caption)
        result_array.append(embeddings)

    return result_array


train_embeddings = run_clip_retrieval(model,  train_df)
val_embeddings = run_clip_retrieval(model,  val_df)
test_embeddings = run_clip_retrieval(model,  test_df)

pickle.dump(train_embeddings, open(EMBEDDINGS_FOLDER + "train_embeddings.pickle", "wb"))
pickle.dump(val_embeddings, open(EMBEDDINGS_FOLDER + "val_embeddings.pickle", "wb"))
pickle.dump(test_embeddings, open(EMBEDDINGS_FOLDER + "test_embeddings.pickle", "wb"))
