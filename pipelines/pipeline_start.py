# Create a function that receives an path to a dataframe, a stable difussion model and an output folder
# and calls the specific pipeline for that model
import pandas as pd
import pickle
import os
from tqdm import tqdm
import shutil


def start_difussion_pipeline(
    input_df_path, image_folder_path, num_output_images, model, output_folder
):
    # Read the dataframe
    f = open(input_df_path, "rb")
    df = pickle.load(f)
    df = df.head(10000)
    df = df.iloc[5300:]

    func = None
    if model == "stable-diffusion-xl":
        from pipelines.stable_difussion_xl_base import stable_difussion_xl_base_pipeline

        func = stable_difussion_xl_base_pipeline
    elif model == "glide":
        from pipelines.glide_pipeline import glide_pipeline

        func = glide_pipeline
    else:
        raise Exception("unkown model pipeline")
    # columsn names
    #'image_id', 'caption', 'file_name', 'clip_scores', 'best_caption',
    #   'bert_embedding', 'bert_embedding_pca', 'cluster_label']
    # For each of the items in the dataframe\

    if os.path.exists(output_folder) == False:
        os.mkdir(output_folder)

    generated_images_df = df[["image_id", "best_caption", "file_name"]]

    generated_images_df["original_image"] = pd.Series(dtype="string")
    generated_images_df["image_1"] = pd.Series(dtype="string")
    generated_images_df["image_2"] = pd.Series(dtype="string")
    generated_images_df["image_3"] = pd.Series(dtype="string")
    generated_images_df["image_4"] = pd.Series(dtype="string")
    generated_images_df["image_5"] = pd.Series(dtype="string")

    for index, row in tqdm(df.iterrows()):
        print(f"Generating {index+1}")
        # Get the prompt and the number of images to generate
        prompt = row["best_caption"]
        image_id = row["image_id"]

        image_output_folder = os.path.join(output_folder, str(image_id))

        if os.path.exists(image_output_folder) == False:
            os.mkdir(image_output_folder)

        func(prompt, num_output_images, image_output_folder)
        shutil.copy(
            os.path.join(image_folder_path, row["file_name"]),
            os.path.join(image_output_folder, "image_6.png"),
        )

        # generated_images_df.loc[index, "original_image"] = os.path.join(
        #    image_output_folder, "image_6.png"
        # )
        # generated_images_df.loc[index, "image_1"] = os.path.join(
        #    image_output_folder, "image_1.png"
        # )
        # generated_images_df.loc[index, "image_2"] = os.path.join(
        #    image_output_folder, "image_2.png"
        # )
        # generated_images_df.loc[index, "image_3"] = os.path.join(
        #    image_output_folder, "image_3.png"
        # )
        # generated_images_df.loc[index, "image_4"] = os.path.join(
        #    image_output_folder, "image_4.png"
        # )
        # generated_images_df.loc[index, "image_5"] = os.path.join(
        #    image_output_folder, "image_5.png"
        # )

    # Save the dataframe
    # generated_images_df.to_csv("generated_images.csv")
