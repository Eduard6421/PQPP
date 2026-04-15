import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
clip_mscoco_retrieval_results_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\retrieval_models\clip\clip_retrieval_results.pickle"
clip_drawbench_retrieval_results_path = fr"C:\Users\User\Desktop\Research\PQPP\retrieval_process\clip\clip_drawbench_retrieval_result.pkl"

best_caption_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\dataset\best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path)
drawbench_prompts_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\drawbench_annotation.csv"
drawbench_prompts = pd.read_csv(drawbench_prompts_path)



drawbench_split_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\drawbench\drawbench_split.csv"
drawbench_split = pd.read_csv(drawbench_split_path)

#test_shuffle_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\shuffle\average_test_shuffle.npy"
#test_shuffle = np.load(test_shuffle_path)

drawbench_split_train = drawbench_split[drawbench_split['split'] == 'train']
drawbench_split_val = drawbench_split[drawbench_split['split'] == 'val'] 
drawbench_split_test = drawbench_split[drawbench_split['split'] == 'test']

drawbench_train_index = np.array(drawbench_split_train['index'].tolist())
drawbench_val_index = np.array(drawbench_split_val['index'].tolist())
drawbench_test_index = np.array(drawbench_split_test['index'].tolist())



# Read both pickle files
with open(clip_mscoco_retrieval_results_path, "rb") as clip_mscoco_retrieval_results_path:
    clip_mscoco_retrieval_results = pickle.load(clip_mscoco_retrieval_results_path)

with open(clip_drawbench_retrieval_results_path, "rb") as clip_drawbench_retrieval_results_path:
    clip_drawbench_retrieval_results = pickle.load(clip_drawbench_retrieval_results_path)


# Print the first element of the list
#print(clip_mscoco_retrieval_results[0])
#print(clip_drawbench_retrieval_results[0].keys())


# retrieved ids from drawbench

drawbench_results = []

for idx,item in enumerate(clip_drawbench_retrieval_results):
    retrieved_items = item["image_ids"]
    drawbench_results.append({idx: retrieved_items})


retrieval_train_gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\retrieval_train_gt.pickle"
retrieval_val_gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\retrieval_val_gt.pickle"
retrieval_test_gt_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\retrieval_test_gt.pickle"

with open(retrieval_train_gt_path, "rb") as f:
    retrieval_train_gt = pickle.load(f)

with open(retrieval_val_gt_path, "rb") as f:
    retrieval_val_gt = pickle.load(f)

with open(retrieval_test_gt_path, "rb") as f:
    retrieval_test_gt = pickle.load(f)



def precision_at_k(retrieved, relevant, k):
    return np.in1d(retrieved[:k], relevant).sum() / k

def reciprocal_rank(retrieved, relevant):
    if(len(relevant) == 0):
        return 0.0

    # Find the first relevant item in the retrieved list and calculate reciprocal rank
    for idx, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1 / idx
    
    # If no relevant item is found in the retrieved list, return 0
    return 0.0




def compute_retrieval_scores(ground_truth_data):


    results = []

    pk_array = []
    rr_array = []

    for item in tqdm(ground_truth_data):
        source = item["source"]
        index = item["index"]
        gt = item["gt"]
        prompt = item["prompt"]
        caption_id = item["caption_id"]

        if source == "drawbench":
            retrieval_result = [int(score) for score in clip_drawbench_retrieval_results[index]['image_ids']]
        elif source =="mscoco":
            retrieval_result = clip_mscoco_retrieval_results[index]
        else:
            raise Exception("Invalid source")


        pk = precision_at_k(retrieval_result, gt, 10)
        rr= reciprocal_rank(retrieval_result, gt)

        results.append({
            'source': source,
            'index': index,
            'precision': pk,
            'reciprocal_rank': rr,
            'prompt': prompt,
            'caption_id': caption_id
        })

        pk_array.append(pk)
        rr_array.append(rr)


        print(f"Index: {index},  Source: {source}, Precision@10: {pk}, Reciprocal Rank: {rr}")
    
    return results

train_results = compute_retrieval_scores(retrieval_train_gt)
val_results = compute_retrieval_scores(retrieval_val_gt)
test_results = compute_retrieval_scores(retrieval_test_gt)


# reorder the columns to
# prompt, source, index, precision, reciprocal_rank
train_results = [{ 'prompt': item['prompt'], 'source': item['source'], 'index': item['index'], 'precision': item['precision'], 'reciprocal_rank': item['reciprocal_rank'], 'caption_id': item['caption_id']} for item in train_results]
val_results = [{ 'prompt': item['prompt'], 'source': item['source'], 'index': item['index'], 'precision': item['precision'], 'reciprocal_rank': item['reciprocal_rank'], 'caption_id': item['caption_id']} for item in val_results]
test_results = [{ 'prompt': item['prompt'], 'source': item['source'], 'index': item['index'], 'precision': item['precision'], 'reciprocal_rank': item['reciprocal_rank'], 'caption_id': item['caption_id']} for item in test_results]


# create pandas dataframes from the results

train_results_df = pd.DataFrame(train_results)
val_results_df = pd.DataFrame(val_results)
test_results_df = pd.DataFrame(test_results)

# Save the results
train_results_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\clip\clip_retrieval_train_results.csv"
val_results_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\clip\clip_retrieval_val_results.csv"
test_results_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\clip\clip_retrieval_test_results.csv"

train_results_df.to_csv(train_results_path)
val_results_df.to_csv(val_results_path)
test_results_df.to_csv(test_results_path)