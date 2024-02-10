import pandas as pd
import numpy as np
import pickle
import torch
import clip
from tqdm import tqdm
from PIL import Image
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"


# dataframe that contains the best caption for each item
best_caption_path = "../../../../dataset/best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path).head(10000)


train_df = best_captions_df.iloc[0:6000]
validation_df = best_captions_df.iloc[6000:8000]
test_df = best_captions_df.iloc[8000:10000]

gt_all_models_path = "../../../../dataset/gt_for_generative_all_models_new.csv"
gt_all_models = pd.read_csv(gt_all_models_path)

# Join train_df with gt_all_models on best_caption

train_df = train_df.merge(
    gt_all_models, how="left", left_on="best_caption", right_on="best_caption"
)

validation_df = validation_df.merge(
    gt_all_models, how="left", left_on="best_caption", right_on="best_caption"
)

test_df = test_df.merge(
    gt_all_models, how="left", left_on="best_caption", right_on="best_caption"
)

model, preprocess = clip.load("ViT-B/32", device=device)


def add_image_links(df):
    df["image_glide_1"] = df["image_id_x"].apply(
        lambda x: f"../../../../output_images/{x}/image_4.png"
    )
    df["image_glide_2"] = df["image_id_x"].apply(
        lambda x: f"../../../../output_images/{x}/image_5.png"
    )
    df["image_clip_1"] = df["image_id_x"].apply(
        lambda x: f"../../../../output_images/{x}/image_7.png"
    )
    df["image_clip_2"] = df["image_id_x"].apply(
        lambda x: f"../../../../output_images/{x}/image_8.png"
    )
    return df


def prepare_image(image):
    if len(np.array(image).shape) != 3:
        image = image.convert("RGB")
    return image


def transform_to_data(df):
    arr = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        glide_image_1 = row["image_glide_1"]
        glide_image_2 = row["image_glide_2"]
        clip_image_1 = row["image_clip_1"]
        clip_image_2 = row["image_clip_2"]

        glide_image_1 = prepare_image(Image.open(glide_image_1))
        glide_image_2 = prepare_image(Image.open(glide_image_2))
        clip_image_1 = prepare_image(Image.open(clip_image_1))
        clip_image_2 = prepare_image(Image.open(clip_image_2))

        glide_image_1 = preprocess(glide_image_1).unsqueeze(0).to(device)
        glide_image_2 = preprocess(glide_image_2).unsqueeze(0).to(device)
        clip_image_1 = preprocess(clip_image_1).unsqueeze(0).to(device)
        clip_image_2 = preprocess(clip_image_2).unsqueeze(0).to(device)

        glide_image_1_features = (
            model.encode_image(glide_image_1).detach().cpu().numpy()
        )
        glide_image_2_features = (
            model.encode_image(glide_image_2).detach().cpu().numpy()
        )
        clip_image_1_features = model.encode_image(clip_image_1).detach().cpu().numpy()
        clip_image_2_features = model.encode_image(clip_image_2).detach().cpu().numpy()

        features_matrix = np.vstack(
            [
                glide_image_1_features,
                glide_image_2_features,
                clip_image_1_features,
                clip_image_2_features,
            ]
        )

        features_transposed = features_matrix.T
        coeff_array = np.corrcoef(features_transposed)
        score = row["score"]
        arr.append((coeff_array, score))

    return arr


train_df = add_image_links(train_df)
validation_df = add_image_links(validation_df)
test_df = add_image_links(test_df)

with open("train_data.pickle", "wb") as f:
    train_data = transform_to_data(train_df)
    pickle.dump(train_data, f)
    del train_data


with open("validation_data.pickle", "wb") as f:
    validation_data = transform_to_data(validation_df)
    pickle.dump(validation_data, f)
    del validation_data

with open("test_data.pickle", "wb") as f:
    test_data = transform_to_data(test_df)
    pickle.dump(test_data, f)
    del test_data
