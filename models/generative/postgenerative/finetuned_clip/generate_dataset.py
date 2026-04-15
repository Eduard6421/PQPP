import pandas as pd
import networkx as nx
import numpy as np
import networkx as nx
import numpy as np
import pickle
import torch
import sys
import os

from PIL import Image
from tqdm import tqdm

# Dynamically add the parent directory of `retrieval_process` to `sys.path`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Now import `longclip` from its location within `retrieval_process`
from postgenerative.LongClip.model import longclip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = longclip.load(fr"C:\Users\User\Desktop\Research\PQPP\models\generative\postgenerative\LongClip\checkpoints\longclip-B.pt", device=device)


selected_model = "sdxl"


train_data = pd.read_csv(fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{selected_model}\{selected_model}_train.csv")
val_data = pd.read_csv(fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{selected_model}\{selected_model}_val.csv")
test_data = pd.read_csv(fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\{selected_model}\{selected_model}_test.csv")




def retrieve_embeddings(model, text):
    text = longclip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)[0]
    return text_features


def extract_pairs_of_scores(data_df):
    mscoco_suffixes = {
        "sdxl" : ["_4","_5"],
        "glide": ["_7","_8"],
    }

    drawbench_suffixes = {
        "sdxl" : ["_4","_5"],
        "glide": ["_6","_7"],
    }


    ms_coco_image_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\output_images"
    drawbench_image_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\drawbench_output"



    arr = []


    # caption_id,caption,score,source

    for index, row in tqdm(data_df.iterrows(), total=len(data_df)):
        caption = row["caption"]
        score = row["score"]
        source = row["source"]

        text_embeddings = retrieve_embeddings(model=model, text=caption)
        text_embeddings = text_embeddings.cpu().numpy()


        if(source == "mscoco"):
            folder_path = os.path.join(ms_coco_image_path, str(row["caption_id"]))
            glide_suffixes = mscoco_suffixes["glide"]
            sdxl_suffixes = mscoco_suffixes["sdxl"]


        elif(source == "drawbench"):
            folder_path = os.path.join(drawbench_image_path, str(row["caption_id"]+10000))
            glide_suffixes = drawbench_suffixes["glide"]
            sdxl_suffixes = drawbench_suffixes["sdxl"]

        else:
            raise Exception("Invalid source")
        
        for suffix in glide_suffixes:
            image_path = os.path.join(folder_path, f"image{suffix}.png")
            current_image = Image.open(image_path)
            current_image = processor(current_image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embeddings = model.encode_image(current_image).cpu().numpy()[0]

            combined_features = np.hstack((text_embeddings, image_embeddings))
            arr.append((row["caption_id"], combined_features, score))

        for suffix in sdxl_suffixes:
            image_path = os.path.join(folder_path, f"image{suffix}.png")
            current_image = Image.open(image_path)
            current_image = processor(current_image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embeddings = model.encode_image(current_image).cpu().numpy()[0]

            combined_features = np.hstack((text_embeddings, image_embeddings))
            arr.append((row["caption_id"], combined_features, score))


    return arr

train_pairs = extract_pairs_of_scores(train_data)
with open(fr"{selected_model}_train_pairs.pkl", "wb") as file:
    pickle.dump(train_pairs, file)

del train_pairs

val_pairs = extract_pairs_of_scores(val_data)
with open(fr"{selected_model}_val_pairs.pkl", "wb") as file:
    pickle.dump(val_pairs, file)
del val_pairs

test_pairs = extract_pairs_of_scores(test_data)
with open(fr"{selected_model}_test_pairs.pkl", "wb") as file:
    pickle.dump(test_pairs, file)
del test_pairs