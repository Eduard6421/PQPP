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


MAX_IMAGES = 25

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# dataframe that contains the best caption for each item
best_caption_path = "../../../../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path).head(10000)


train_df = best_captions_df.iloc[0:6000]
validation_df = best_captions_df.iloc[6000:8000]
test_df = best_captions_df.iloc[8000:10000]

ground_truth_map = "../../../../dataset/merged_retrieval_gt_new.pickle"
ground_truth = pickle.load(open(ground_truth_map, "rb"))


def split_obj_to_dataset(obj):
    train = []
    validation = []
    test = []

    for i in range(6000):
        train.append(obj[i])

    for i in range(6000, 8000):
        validation.append(obj[i])

    for i in range(8000, 10000):
        test.append(obj[i])

    return train, validation, test


clip_retrieved_items = "../../../../retrieval_models/clip/clip_retrieval_results.pickle"
clip_retrieved = pickle.load(open(clip_retrieved_items, "rb"))

blip2_retrieved_items = (
    "../../../../retrieval_models/blip2/blip2_retrieval_results.pickle"
)
blip2_retrieved = pickle.load(open(blip2_retrieved_items, "rb"))


ground_truth_train, ground_truth_validation, ground_truth_test = split_obj_to_dataset(
    ground_truth
)

train_clip_preds, validation_clip_preds, test_clip_preds = split_obj_to_dataset(
    clip_retrieved
)

train_blip2_preds, validation_blip2_preds, test_blip2_preds = split_obj_to_dataset(
    blip2_retrieved
)


def id_to_embedding(image_id):
    base_image_folder = "../../../../dataset/train2017/train2017/"
    image_id = str(image_id).zfill(12) + ".jpg"
    image = Image.open(base_image_folder + image_id)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).detach().cpu().numpy()
    return image_features


def extract_pairs_of_scores(queries_df, clip_results, blip2_results, true_gt):
    arr = []

    queries_df = queries_df.reset_index()

    # Iterate over the queries_df line by line

    for idx in tqdm(range(queries_df.shape[0])):
        text = queries_df["best_caption"][idx]
        # image_id = queries_df["image_id"][idx]
        text = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_embeddings = (
                model.encode_text(text)[0].unsqueeze(0).detach().cpu().numpy()
            )

        query_clip_method_results = clip_results[idx][:MAX_IMAGES]
        query_blip2_method_results = blip2_results[idx][:MAX_IMAGES]
        query_gt = true_gt[idx]

        for item in query_clip_method_results:
            if item in query_gt:
                score = 1
            else:
                score = 0
            image_features = id_to_embedding(item)
            combined_features = np.hstack((text_embeddings, image_features))
            arr.append((combined_features, score))

        for item in query_blip2_method_results:
            if item in query_gt:
                score = 1
            else:
                score = 0
            image_features = id_to_embedding(item)
            combined_features = np.hstack((text_embeddings, image_features))
            arr.append((combined_features, score))

    return arr


train_dataset = extract_pairs_of_scores(
    train_df, train_clip_preds, train_blip2_preds, ground_truth_train
)

validation_dataset = extract_pairs_of_scores(
    validation_df,
    validation_clip_preds,
    validation_blip2_preds,
    ground_truth_validation,
)

test_dataset = extract_pairs_of_scores(
    test_df, test_clip_preds, test_blip2_preds, ground_truth_test
)

pickle.dump(train_dataset, open("./train_dataset.pickle", "wb"))
pickle.dump(validation_dataset, open("./validation_dataset.pickle", "wb"))
pickle.dump(test_dataset, open("./test_dataset.pickle", "wb"))

"""

train_array = [(train_df["image_id_y"][i], train_df["caption"][i]) for i in range(6000)]
validation_array = [
    (validation_df["image_id_y"][i], validation_df["caption"][i]) for i in range(2000)
]
test_array = [(test_df["image_id_y"][i], test_df["caption"][i]) for i in range(2000)]

print(train_array[0])

"""
"""

class CustomCLIPModel(torch.nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPModel, self).__init__()
        self.clip_model = clip_model
        # Assuming the embedding size of CLIP is 512. Adjust according to your CLIP model version.
        self.regressor = torch.nn.Linear(1024, 1)

    def forward(self, images, input_ids, attention_mask):
        # Concatenate or combine the features in a meaningful way for your task
        combined_features = (
            text_features + image_features
        )  # This is a simplification; consider more complex fusion strategies

        # Pass through the regressor to get the score
        score = self.regressor(combined_features)

        return score
"""
