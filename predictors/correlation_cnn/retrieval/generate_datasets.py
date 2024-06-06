import pandas as pd
import numpy as np
import pickle
import torch
import clip
from tqdm import tqdm
from PIL import Image
import os
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"


# dataframe that contains the best caption for each item
best_caption_path = "../../../../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path).head(10000)


NUM_MAX_IMAGES = 25


train_df = best_captions_df.iloc[0:6000]
validation_df = best_captions_df.iloc[6000:8000]
test_df = best_captions_df.iloc[8000:10000]

gt_all_mrr_scores = "../../../dataset/avg_scores_mrr_new.pickle"
with open(gt_all_mrr_scores, "rb") as f:
    gt_all_mrr_scores = pickle.load(f)
parsed_scores = []
for i in range(10000):
    parsed_scores.append(gt_all_mrr_scores[i])
train_array_scores = parsed_scores[:6000]
validation_array_scores = parsed_scores[6000:8000]
test_array_scores = parsed_scores[8000:10000]


model, preprocess = clip.load("ViT-B/32", device=device)


clip_predictions_path = "../../../dataset/clip/clip_retrieval_results.pickle"
blip2_predictions_path = "../../../dataset/blip2_retrieval_results.pickle"


with open(clip_predictions_path, "rb") as f:
    clip_predictions = pickle.load(f)

with open(blip2_predictions_path, "rb") as f:
    blip_predictions = pickle.load(f)

listed_clip_preds = []
listed_blip2_preds = []
for i in range(10000):
    listed_clip_preds.append(clip_predictions[i])
    listed_blip2_preds.append(blip_predictions[i])

train_array_predictions_clip = listed_clip_preds[:6000]
validation_array_predictions_clip = listed_clip_preds[6000:8000]
test_array_predictions_clip = listed_clip_preds[8000:10000]

train_array_predictions_blip = listed_blip2_preds[:6000]
validation_array_predictions_blip = listed_blip2_preds[6000:8000]
test_array_predictions_blip = listed_blip2_preds[8000:10000]


def prepare_image(image):
    if len(np.array(image).shape) != 3:
        image = image.convert("RGB")
    return image


def generate_corr_matrices(image_ids):
    base_folder = "../../../../dataset/train2017/train2017/"

    feature_arrays = []
    for image_id in image_ids:
        image_id = str(image_id).zfill(12) + ".jpg"
        full_image_path = base_folder + image_id
        image = Image.open(full_image_path)
        image = prepare_image(image)
        image = preprocess(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image).detach().cpu().numpy()
        feature_arrays.append(image_features)

    features_matrix = np.vstack(feature_arrays)
    features_transposed = features_matrix.T

    corr_matrix = np.corrcoef(features_transposed)
    return corr_matrix


def generate_data(df, gt, preds_clip, preds_blip):
    data = []

    for i in tqdm(range(len(preds_clip))):
        score = gt[i]
        clip_img_ids = preds_clip[i][:NUM_MAX_IMAGES]
        blip_img_ids = preds_blip[i][:NUM_MAX_IMAGES]

        corr_matrices_clip = generate_corr_matrices(clip_img_ids)
        corr_matrices_blip = generate_corr_matrices(blip_img_ids)
        # stack the two 512x512 matrices so that it is 2x512x512
        corr_matrices = np.stack((corr_matrices_clip, corr_matrices_blip))

        data.append((corr_matrices, score))

    return data


train_data = generate_data(
    train_df,
    train_array_scores,
    train_array_predictions_clip,
    train_array_predictions_blip,
)
validation_data = generate_data(
    validation_df,
    validation_array_scores,
    validation_array_predictions_clip,
    validation_array_predictions_blip,
)
test_data = generate_data(
    test_df, test_array_scores, test_array_predictions_clip, test_array_predictions_blip
)


with open("./train_data_clip_mrr.pickle", "wb") as f:
    pickle.dump(train_data, f)

with open("./validation_data_clip_mrr.pickle", "wb") as f:
    pickle.dump(validation_data, f)

with open("./test_data_clip_mrr.pickle", "wb") as f:
    pickle.dump(test_data, f)
