import pandas as pd
import gensim.downloader as api
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import multiprocessing
import numpy as np
import concurrent.futures
import networkx as nx
import numpy as np
from nltk.corpus import stopwords
import pickle
from tqdm import tqdm
import nltk
import concurrent.futures
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# dataframe that contains the best caption for each item
best_caption_path = "../../../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path).head(10000)


train_df = best_captions_df.iloc[0:6000]
validation_df = best_captions_df.iloc[6000:8000]
test_df = best_captions_df.iloc[8000:10000]

individual_scores_path = (
    "../../../dataset/generative_models_scores/gt_for_generative_individual_new.csv"
)
individual_scores = pd.read_csv(individual_scores_path)


def extract_pairs_of_scores(annotation_df, scores_df):
    suffixes = ["_4", "_5", "_7", "_8"]

    scores_map = {}
    arr = []
    for index, row in scores_df.iterrows():
        score = row["score"]
        image_id = row["image_id"]
        scores_map[image_id] = score

    # iterate throug hte dataset
    for index, row in tqdm(annotation_df.iterrows(), total=len(annotation_df)):
        text = row["best_caption"]
        text = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_embeddings = (
                model.encode_text(text)[0].unsqueeze(0).detach().cpu().numpy()
            )
        for suffix in suffixes:
            base_image_id = str(row["image_id"])
            current_image_id = str(row["image_id"]) + suffix
            current_image_path = (
                f"../../../../output_images/{base_image_id}/image{suffix}.png"
            )
            current_image = Image.open(current_image_path)
            current_image = preprocess(current_image).unsqueeze(0).to(device)
            individual_score = scores_map[current_image_id]

            with torch.no_grad():
                image_features = (
                    model.encode_image(current_image).detach().cpu().numpy()
                )

            # stack the two matrices of size 2x512 together
            combined_features = np.hstack((text_embeddings, image_features))
            arr.append((base_image_id, combined_features, individual_score))
    return arr


train_dataset = extract_pairs_of_scores(
    train_df,
    individual_scores,
)

validation_dataset = extract_pairs_of_scores(
    validation_df,
    individual_scores,
)

test_dataset = extract_pairs_of_scores(
    test_df,
    individual_scores,
)

pickle.dump(train_dataset, open("./train_dataset.pickle", "wb"))
pickle.dump(validation_dataset, open("./validation_dataset.pickle", "wb"))
pickle.dump(test_dataset, open("./test_dataset.pickle", "wb"))
