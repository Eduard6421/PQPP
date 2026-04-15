import pandas as pd
import numpy as np
import numpy as np
import pickle

import os
import sys
import torch


from tqdm import tqdm
from PIL import Image

MAX_IMAGES = 25

device = "cuda" if torch.cuda.is_available() else "cpu"
# Dynamically add the parent directory of `retrieval_process` to `sys.path`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Now import `longclip` from its location within `retrieval_process`

from postretrieval.LongClip.model import longclip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = longclip.load(fr"C:\Users\User\Desktop\Research\PQPP\models\generative\postgenerative\LongClip\checkpoints\longclip-B.pt", device=device)

train_retrieval_gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\retrieval_train_gt.pickle"
val_retrieval_gt_path  = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\retrieval_val_gt.pickle"
test_retrieval_gt_path  = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\retrieval_test_gt.pickle"

ground_truth_train = pickle.load(open(train_retrieval_gt_path , "rb"))
ground_truth_val= pickle.load(open(val_retrieval_gt_path , "rb"))
ground_truth_test = pickle.load(open(test_retrieval_gt_path , "rb"))

clip_mscoco_retrieval_results_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\retrieval_models\clip\clip_retrieval_results.pickle"
clip_drawbench_retrieval_results_path = fr"C:\Users\User\Desktop\Research\PQPP\retrieval_process\clip\clip_drawbench_retrieval_result.pkl"



# Read both pickle files
with open(clip_mscoco_retrieval_results_path, "rb") as f:
    clip_mscoco_retrieval_results = pickle.load(f)

with open(clip_drawbench_retrieval_results_path, "rb") as f:
    clip_drawbench_retrieval_results = pickle.load(f)

def generate_text_embedding(text):
    text = longclip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)[0]
    return text_features

def id_to_embedding(image_id):
    base_image_folder = "C:/Users/User/Desktop/Research/stable-prompt-pred/dataset/train2017/train2017/"
    image_id = str(image_id).zfill(12) + ".jpg"
    image = Image.open(base_image_folder + image_id)
    image = processor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).detach().cpu().numpy()
    return image_features




def generate_retrieval_dataset(clip_mscoco_retrieval_results,clip_drawbench_retrieval_results, ground_truth):

    arr = []

    for item in tqdm(ground_truth):
        source = item['source']
        index = item["index"]        

        if source == "drawbench":
            retrieval_result = [int(image_id) for image_id in clip_drawbench_retrieval_results[index]['image_ids']]
        elif source =="mscoco":
            retrieval_result = clip_mscoco_retrieval_results[index]
        else:
            raise Exception("Invalid source")

        gt = item['gt']

        text_embeddings = generate_text_embedding(item['prompt'])

        for image_id in retrieval_result[:MAX_IMAGES]:
            if image_id in gt:
                score = 1
            else:
                score = 0

            image_features = id_to_embedding(image_id)



            combined_features = np.hstack((text_embeddings.detach().cpu().numpy(), image_features.squeeze(0)))
            arr.append((combined_features, score))

    return arr

train_data = generate_retrieval_dataset(clip_mscoco_retrieval_results,clip_drawbench_retrieval_results, ground_truth_train)
with open("clip_train_dataset.pickle", "wb") as file:
    pickle.dump(train_data, file)

val_data = generate_retrieval_dataset(clip_mscoco_retrieval_results,clip_drawbench_retrieval_results, ground_truth_val)
with open("clip_val_dataset.pickle", "wb") as file:
    pickle.dump(val_data, file)

test_data = generate_retrieval_dataset(clip_mscoco_retrieval_results,clip_drawbench_retrieval_results, ground_truth_test)
with open("clip_test_dataset.pickle", "wb") as file:
    pickle.dump(test_data, file)