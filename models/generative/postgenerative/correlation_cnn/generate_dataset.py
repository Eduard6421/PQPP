import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
from PIL import Image
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"


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



def prepare_image(image):
    if len(np.array(image).shape) != 3:
        image = image.convert("RGB")
    return image



def retrieve_embeddings(model, text):
    text = longclip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)[0]
    return text_features


def extract_pairs_of_scores(data_df):
    mscoco_suffixes = ["_4", "_5", "_7", "_8"]
    drawbench_suffixes = ["_4","_5","_6","_7"]

    ms_coco_image_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\output_images"
    drawbench_image_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\drawbench_output"


    arr = []


    # caption_id,caption,score,source

    for index, row in tqdm(data_df.iterrows(), total=len(data_df)):
        score = row["score"]
        source = row["source"]


        if(source == "mscoco"):
            folder_path = os.path.join(ms_coco_image_path, str(row["caption_id"]))
            suffixes = mscoco_suffixes

        elif(source == "drawbench"):
            folder_path = os.path.join(drawbench_image_path, str(row["caption_id"]+10000))
            suffixes = drawbench_suffixes

        else:
            raise Exception("Invalid source")
        
        img_arr = []
        
        for suffix in suffixes:
            image_path = os.path.join(folder_path, f"image{suffix}.png")
            current_image = Image.open(image_path)
            current_image = processor(current_image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embeddings = model.encode_image(current_image).cpu().numpy()[0]
            img_arr.append(image_embeddings)

        combined_features = np.vstack(img_arr)
        features_transposed = combined_features.T
        coeff_array = np.corrcoef(features_transposed)
        arr.append((coeff_array, score))


    return arr


train_data = extract_pairs_of_scores(train_data)
with open("correlation_cnn_train_data.pkl", "wb") as file:
    pickle.dump(train_data, file)

val_data = extract_pairs_of_scores(val_data)
with open("correlation_cnn_val_data.pkl", "wb") as file:
    pickle.dump(val_data, file)

test_data = extract_pairs_of_scores(test_data)
with open("correlation_cnn_test_data.pkl", "wb") as file:
    pickle.dump(test_data, file)