import os
import pickle
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F


from tqdm import tqdm
from lavis.models import load_model_and_preprocess


MSCOCO_EMBEDDINGS_FOLDER = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\retrieval_models\blip2\blip2_image_embeddings"
DRAWBENCH_EMBEDDINGS_FOLDER = fr"C:\Users\User\Desktop\Research\PQPP\retrieval_process\blip2\drawbench_embeddings"

TRAIN_QUERIES_PATH = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_train.csv"
VAL_QUERIES_PATH = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_val.csv"
TEST_QUERIES_PATH = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\ground_truth\average\average_test.csv"


DRAWBENCH_PROMPTS_PATH = fr"C:\Users\User\Desktop\Research\PQPP\dataset\drawbench_annotation.csv"


device = "cpu"

model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_feature_extractor",
    model_type="pretrain_vitL",
    is_eval=True,
    device=device,
)


def load_image_embeddings(source, embeddings_folder):
    embeddings = []
    image_ids = []
    for file in tqdm(os.listdir(embeddings_folder), desc=fr"Loading embeddings from {source}"):
        with open(os.path.join(embeddings_folder, file), "rb") as f:
            data = pickle.load(f)
            embeddings += data["embeddings"]
            image_ids += data["image_ids"]

    sources = [source] * len(embeddings)
    return embeddings, image_ids, sources

def generate_text_embedding(text):
    text_processed = [txt_processors["eval"](text)]
    sample = {"text_input": text_processed}

    with torch.no_grad():
        text_embeddings = model.extract_features(sample, mode="text")
    cls_embedding = text_embeddings.text_embeds[:, 0, :]

    return cls_embedding

def generate_text_embeddings(texts):
    text_processed = [txt_processors["eval"](text) for text in texts]
    sample = {"text_input": text_processed}

    with torch.no_grad():
        text_embeddings = model.extract_features(sample, mode="text")
    cls_embedding = text_embeddings.text_embeds[:, 0, :]

    return cls_embedding




def perform_retrieval(query_embedding, image_embeddings, image_ids, image_sources):
    # Step 1: Transform the second array into a tensor
    if isinstance(image_embeddings, list):
        # Convert list of embeddings to a tensor
        image_embeddings = torch.stack([
            torch.tensor(emb) if not isinstance(emb, torch.Tensor) else emb
            for emb in image_embeddings
        ])  # Shape: [num_images, 32, 768]
    elif isinstance(image_embeddings, np.ndarray):
        image_embeddings = torch.from_numpy(image_embeddings)  # Convert numpy array to tensor
    elif isinstance(image_embeddings, torch.Tensor):
        pass  # Already a tensor
    else:
        raise ValueError("image_embeddings must be a list, numpy array, or torch tensor")

    # Ensure all tensors are on the same device
    device = query_embedding.device
    image_embeddings = image_embeddings.to(device)

    # Step 2: Compute cosine similarities
    # Ensure query_embedding is of shape [1, 768]
    if query_embedding.dim() == 2 and query_embedding.size(0) == 1:
        pass  # Already correct shape
    elif query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)  # Shape: [1, 768]
    else:
        raise ValueError(f"Unexpected query_embedding shape: {query_embedding.shape}")

    # Reshape query_embedding to [1, 1, 768] for broadcasting
    query_embedding = query_embedding.unsqueeze(1)  # Shape: [1, 1, 768]

    # Compute cosine similarity between query_embedding and each image embedding
    # image_embeddings shape: [num_images, 32, 768]
    # query_embedding shape: [1, 1, 768] -> broadcasted to [num_images, 32, 768]
    cos_sim = F.cosine_similarity(image_embeddings, query_embedding, dim=-1)  # Shape: [num_images, 32]

    # Step 3: Compute the maximum over the 32 similarities per image
    max_cos_sim, _ = cos_sim.max(dim=1)  # Shape: [num_images]

    # Step 4: Sort the similarities in descending order
    sorted_similarities, sorted_indices = torch.sort(max_cos_sim, descending=True)  # Shape: [num_images]



    # Step 5: Return the sorted values, image_ids, and image_sources
    sorted_image_ids = [image_ids[idx] for idx in sorted_indices]
    sorted_image_sources = [image_sources[idx] for idx in sorted_indices]


    # If needed, convert tensors to CPU before returning
    sorted_similarities = sorted_similarities.cpu()


    # Return the sorted similarities, image IDs, and sources
    return sorted_similarities, sorted_image_ids, sorted_image_sources



def compute_retrieval_scores(df, image_embeddings, image_ids, image_sources):
    captions = df["caption"].tolist()

    retrieval_results= []
    # for caption in captions:

    for caption in tqdm(captions):
        caption_embeddings = generate_text_embedding(caption)
        scores, image_ids, image_sources = perform_retrieval(caption_embeddings, image_embeddings, image_ids, image_sources)
        retrieval_results.append({
            'scores': scores,
            'image_ids': image_ids,
            'image_sources': image_sources
        })
            
    return retrieval_results
        
# Load the image embeddings for each of them
mscoco_image_embeddings, mscoco_image_ids, mscoco_image_sources = load_image_embeddings("MS_COCO", MSCOCO_EMBEDDINGS_FOLDER)
#drawbench_image_embeddings, drawbench_image_ids, drawbench_image_sources = load_image_embeddings("Drawbench", DRAWBENCH_EMBEDDINGS_FOLDER)

all_images = np.array(mscoco_image_embeddings )
all_image_ids = np.array(mscoco_image_ids)
all_image_sources = np.array(mscoco_image_sources)

# Load train/val/test dataframes
train_df = pd.read_csv(TRAIN_QUERIES_PATH)
val_df = pd.read_csv(VAL_QUERIES_PATH)
test_df = pd.read_csv(TEST_QUERIES_PATH)



train_retrieval_results = compute_retrieval_scores(train_df, image_embeddings=all_images, image_ids=all_image_ids, image_sources=all_image_sources)
pickle.dump('./train_blip_retrieval_results.pkl')


#val_retrieval_results = compute_retrieval_scores(val_df, image_embeddings=all_images, image_ids=all_image_ids, image_sources=all_image_sources)
#pickle.dump('./val_blip_retrieval_results.pkl')


#test_retrieval_results = compute_retrieval_scores(train_df, image_embeddings=all_images, image_ids=all_image_ids, image_sources=all_image_sources)
#pickle.dump('./test_blip_retrieval_results.pkl')