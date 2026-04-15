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

from postretrieval.LongClip.model import longclip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, processor = longclip.load(fr"C:\Users\User\Desktop\Research\PQPP\models\generative\postgenerative\LongClip\checkpoints\longclip-B.pt", device=device)

train_retrieval_gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\average\average_train.csv"
val_retrieval_gt_path  = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\average\average_val.csv"
test_retrieval_gt_path  = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\average\average_test.csv"

ground_truth_train = pd.read_csv(train_retrieval_gt_path)
ground_truth_val= pd.read_csv(val_retrieval_gt_path)
ground_truth_test = pd.read_csv(test_retrieval_gt_path)

clip_mscoco_retrieval_results_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\retrieval_models\clip\clip_retrieval_results.pickle"
clip_drawbench_retrieval_results_path = fr"C:\Users\User\Desktop\Research\PQPP\retrieval_process\clip\clip_drawbench_retrieval_result.pkl"

blip2_mscoco_retrieval_results_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\retrieval_models\blip2\blip2_retrieval_results.pickle"
blip2_drawbench_retrieval_results_path = fr"C:\Users\User\Desktop\Research\PQPP\retrieval_process\blip2\blip2_drawbench_retrieval_result.pkl"



def retrieve_embeddings(model, text):
    text = longclip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)[0]
    return text_features

NUM_MAX_IMAGES = 25


def prepare_image(image):
    if len(np.array(image).shape) != 3:
        image = image.convert("RGB")
    return image


def generate_corr_matrices(image_ids):

    base_folder = "C:/Users/User/Desktop/Research/stable-prompt-pred/dataset/train2017/train2017/"
    feature_arrays = []
    for image_id in image_ids:
        image_id = str(image_id).zfill(12) + ".jpg"
        full_image_path = base_folder + image_id
        image = Image.open(full_image_path)
        image = prepare_image(image)
        image = processor(image).unsqueeze(0).to(device)
        image_features = model.encode_image(image).detach().cpu().numpy()
        feature_arrays.append(image_features)

    features_matrix = np.vstack(feature_arrays)
    features_transposed = features_matrix.T

    corr_matrix = np.corrcoef(features_transposed)

    
    if np.isnan(corr_matrix).any():
        raise Exception("NAN values found in correlation matrix")

    return corr_matrix


def generate_data(blip2_mscoco_retrieval_results, clip_mscoco_retrieval_results, blip2_drawbench_retrieval_results,clip_drawbench_retrieval_results, ground_truth):
    data = []

    for index, row in tqdm(ground_truth.iterrows(), total=len(ground_truth)):

        source = row['source']
        index = int(row["index"])

        if source == "drawbench":
            retrieval_result_clip = [int(image_id) for image_id in clip_drawbench_retrieval_results[index]['image_ids']]
            retrieval_result_blip2 = [int(image_id) for image_id in blip2_drawbench_retrieval_results[index]['image_ids']]            
        elif source =="mscoco":
            retrieval_result_clip = clip_mscoco_retrieval_results[index]
            retrieval_result_blip2 = blip2_mscoco_retrieval_results[index]
        else:
            raise Exception("Invalid source")

        score = row['precision']    

        clip_img_ids = retrieval_result_clip[:NUM_MAX_IMAGES]
        blip_img_ids = retrieval_result_blip2[:NUM_MAX_IMAGES]

        corr_matrices_clip = generate_corr_matrices(clip_img_ids)
        corr_matrices_blip = generate_corr_matrices(blip_img_ids)
            # stack the two 512x512 matrices so that it is 2x512x512
        corr_matrices = np.stack((corr_matrices_clip, corr_matrices_blip))

        data.append((corr_matrices, score))

    return data



blip2_mscoco_retrieval_results = pickle.load(open(blip2_mscoco_retrieval_results_path, "rb"))
clip_mscoco_retrieval_results = pickle.load(open(clip_mscoco_retrieval_results_path, "rb"))

blip2_drawbench_retrieval_results = pickle.load(open(blip2_drawbench_retrieval_results_path, "rb"))
clip_drawbench_retrieval_results = pickle.load(open(clip_drawbench_retrieval_results_path, "rb"))


#print([image_id for image_id in clip_drawbench_retrieval_results[29]['image_ids']])
#print(clip_drawbench_retrieval_results[29])
'''
train_data = generate_data(blip2_mscoco_retrieval_results, clip_mscoco_retrieval_results, blip2_drawbench_retrieval_results, clip_drawbench_retrieval_results, ground_truth_train)
with open("corr_train_dataset.pickle", "wb") as file:
    pickle.dump(train_data, file)
del train_data

val_data = generate_data(blip2_mscoco_retrieval_results, clip_mscoco_retrieval_results, blip2_drawbench_retrieval_results, clip_drawbench_retrieval_results, ground_truth_val)
with open("corr_val_dataset.pickle", "wb") as file:
    pickle.dump(val_data, file)

del val_data
'''
test_data = generate_data(blip2_mscoco_retrieval_results, clip_mscoco_retrieval_results, blip2_drawbench_retrieval_results, clip_drawbench_retrieval_results, ground_truth_test)
with open("corr_test_dataset.pickle", "wb") as file:
    pickle.dump(test_data, file)

del test_data