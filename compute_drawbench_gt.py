import os
import pickle
import pandas as pd
drawbench_gt_file = fr"D:\drawbench_images"
mscoco_gt_file = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\dataset\merged_retrieval_gt_new.pickle"


best_caption_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\dataset\best_captions_df.pickle"
best_captions_df = pd.read_pickle(best_caption_path)
# Write script which iterates from 0 to 200

# get all folders in the directory drawbench_gt_file

folders = os.listdir(drawbench_gt_file)

# folders are of shape <NUMBER>_<TEXT>
# I want you to sort them by the <NUMBER> part

folders = sorted(folders, key=lambda x: int(x.split("_")[0]))

# Now iterate over the folders

drawbench_gt = {}

for folder in folders:
    matches = []
    #extract the number
    number = folder.split("_")[0]
    # list all files in folder
    files = os.listdir(os.path.join(drawbench_gt_file, folder))
    for file in files:
        second_number = file.split("_")[2].split(".")[0]
        # undo zfill(12)
        second_number = int(second_number)
        matches.append(second_number)

    drawbench_gt[int(number)] = matches

with open(mscoco_gt_file, "rb") as f:
    mscoco_gt = pickle.load(f)


#Join the two objects
#merged_gt = mscoco_gt + drawbench_gt
#print(mscoco_gt.keys())
#print(drawbench_gt.keys())
#print(merged_gt.keys())


import numpy as np
import pandas as pd
# train shuffle. all the shuffles are actually the same

np.random.seed(14)

drawbench_data = fr"C:\Users\User\Desktop\Research\PQPP\dataset\drawbench_annotation.csv"
drawbench_data = pd.read_csv(drawbench_data)

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


train_data = []
for i in range(6000):
    train_data.append({
        "source": "mscoco",
        "index": i,
        "gt": mscoco_gt[i],
        "prompt": best_captions_df.iloc[i]["best_caption"],
        "caption_id": best_captions_df.iloc[i]["id"]
    })

for i in drawbench_train_index:
    train_data.append({
        "source": "drawbench",
        "index": int(i),
        "gt": drawbench_gt[int(i)],
        "prompt": drawbench_data.iloc[int(i)]["Prompt"],
        "caption_id": int(i),

    })

val_data = []
for i in range(6000, 8000):
    val_data.append({
        "source": "mscoco",
        "index": i,
        "gt": mscoco_gt[i],
        "prompt": best_captions_df.iloc[i]["best_caption"],
        "caption_id": best_captions_df.iloc[i]["id"],
    })

for i in drawbench_val_index:
    val_data.append({
        "source": "drawbench",
        "index": int(i),
        "gt": drawbench_gt[int(i)],
        "prompt": drawbench_data.iloc[int(i)]["Prompt"],
        "caption_id": int(i),        
    })

test_data = []
for i in range(8000, 10000):
    test_data.append({
        "source": "mscoco",
        "index": i,
        "gt": mscoco_gt[i],
        "prompt": best_captions_df.iloc[i]["best_caption"],
        "caption_id": best_captions_df.iloc[i]["id"]             
    })

for i in drawbench_test_index:
    test_data.append({
        "source": "drawbench",
        "index": int(i),
        "gt": drawbench_gt[int(i)],
        "prompt": drawbench_data.iloc[int(i)]["Prompt"],
        "caption_id": int(i),        
    })

train_shuffle = np.load(fr"C:\Users\User\Desktop\Research\PQPP\dataset\shuffle\train_shuffle.npy")
val_shuffle = np.load(fr"C:\Users\User\Desktop\Research\PQPP\dataset\shuffle\val_shuffle.npy")
test_shuffle = np.load(fr"C:\Users\User\Desktop\Research\PQPP\dataset\shuffle\test_shuffle.npy")

train_data = [train_data[i] for i in train_shuffle]
val_data = [val_data[i] for i in val_shuffle]
test_data = [test_data[i] for i in test_shuffle]

train_data_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\retrieval_train_gt.pickle"
val_data_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\retrieval_val_gt.pickle"
test_data_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\retrieval_test_gt.pickle"

with open(train_data_path, "wb") as f:
    pickle.dump(train_data, f)

with open(val_data_path, "wb") as f:
    pickle.dump(val_data, f)

with open(test_data_path, "wb") as f:
    pickle.dump(test_data, f)
