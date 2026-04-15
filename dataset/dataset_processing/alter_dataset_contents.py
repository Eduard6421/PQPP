import pandas as pd
import numpy as np
import os 

np.random.seed(14)

task = "average" # Available tasks : sdxl , glide, average

# File paths with raw string notation
drawbench_csv_file_path = fr'C:\Users\User\Desktop\Research\PQPP\dataset\generative\drawbench\drawbench_generative_task_{task}_score_gt.csv'
mscoco_csv_file_path = fr'C:\Users\User\Desktop\Research\PQPP\dataset\generative\mscoco\mscoco_generative_task_{task}_score_gt.csv'
split_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\generative\drawbench\drawbench_split.csv"

def get_or_create_drawbench_data_split(split_path, drawbench_csv_file_path):
    try:
        # Attempt to load existing splits
        splits = pd.read_csv(split_path, encoding='utf-8')
        print("Splits loaded from file")
        return splits
    except FileNotFoundError:
        print("Creating new stratified splits")

        data = pd.read_csv(drawbench_csv_file_path, encoding='utf-8')

        total_train_size = 80
        total_val_size = 40
        total_test_size = 80
        total_desired = total_train_size + total_val_size + total_test_size  # Should be 200


        # Get the list of unique classes
        classes = data['Category'].unique()
        num_classes = len(classes)

        # Initialize lists to hold indices for each split
        train_indices = []
        val_indices = []
        test_indices = []

        # Iterate over each class to assign samples
        for cls in classes:
            # Get all indices of the current class
            class_indices = data[data['Category'] == cls].index.tolist()
            np.random.shuffle(class_indices)
            n_samples = len(class_indices)

            # Ensure at least one sample per split if possible
            if n_samples >= 3:
                # Proportional split based on desired totals
                train_prop = total_train_size / total_desired
                val_prop = total_val_size / total_desired
                test_prop = total_test_size / total_desired

                # Calculate per-class split sizes
                train_size = max(1, int(round(train_prop * n_samples)))
                val_size = max(1, int(round(val_prop * n_samples)))
                test_size = n_samples - train_size - val_size

                # Adjust sizes if necessary to ensure at least one sample in each split
                if test_size == 0:
                    test_size = 1
                    if train_size > val_size:
                        train_size -= 1
                    else:
                        val_size -= 1

                # Correct any rounding issues
                while train_size + val_size + test_size > n_samples:
                    # Reduce the largest split
                    max_size = max(train_size, val_size, test_size)
                    if train_size == max_size and train_size > 1:
                        train_size -= 1
                    elif val_size == max_size and val_size > 1:
                        val_size -= 1
                    elif test_size == max_size and test_size > 1:
                        test_size -= 1
                    else:
                        break  # Cannot reduce further

                while train_size + val_size + test_size < n_samples:
                    # Increase the smallest split
                    min_size = min(train_size, val_size, test_size)
                    if train_size == min_size:
                        train_size += 1
                    elif val_size == min_size:
                        val_size += 1
                    else:
                        test_size += 1

            elif n_samples == 2:
                # Assign one sample to train and one to test
                train_size = 1
                val_size = 0
                test_size = 1
            else:
                raise ValueError("Class has fewer than 2 samples")

            # Assign indices to splits
            train_indices.extend(class_indices[:train_size])
            val_indices.extend(class_indices[train_size:train_size + val_size])
            test_indices.extend(class_indices[train_size + val_size:train_size + val_size + test_size])

        # Compile the splits into a DataFrame
        split_data = pd.DataFrame({
            'index': train_indices + val_indices + test_indices,
            'split': ['train'] * len(train_indices) + ['val'] * len(val_indices) + ['test'] * len(test_indices)
        })

        # Merge split information back into the original data
        split_data = split_data.merge(data.reset_index(), on='index')

        # Save the split data to CSV
        split_data.to_csv(split_path, index=False, columns=['index', 'split'], encoding='utf-8')
        print("New stratified splits created and saved to file")

        # Return the split information
        return split_data[['index', 'split']]
    

def generate_joint_data_splits(drawbench_csv_file_path, mscoco_csv_file_path, splits):


    # MSCOCO

    # ms_coco csv columns: image_id,best_caption,score
    mscoco_dataset_df = pd.read_csv(mscoco_csv_file_path, encoding='utf-8')

    # rename to caption_id, caption, score with the rename function
    mscoco_dataset_df.rename(columns={'image_id':'caption_id', 'best_caption':'caption'}, inplace=True) 

    # add a column to the dataframe with the name 'source' and fill it with 'mscoco'
    mscoco_dataset_df['source'] = 'mscoco'


    # Select the first 6000 rows from the mscoco dataframe and store it in train
    train_mscoco = mscoco_dataset_df.iloc[:6000]

    # Select the next 2000 rows from the mscoco dataframe and store it in val
    val_mscoco = mscoco_dataset_df.iloc[6000:8000]

    # Select the last 2000 rows from the mscoco dataframe and store it in test
    test_mscoco = mscoco_dataset_df.iloc[8000:10000]

    # Reset the indexes
    train_mscoco.reset_index(drop=True, inplace=True)
    val_mscoco.reset_index(drop=True, inplace=True)
    test_mscoco.reset_index(drop=True, inplace=True)



    # Drawbench

    # drawbench csv columns: Prompts,Category,score
    drawbench_dataset_df = pd.read_csv(drawbench_csv_file_path, encoding='utf-8')

    # Rename Prompts to caption
    drawbench_dataset_df.rename(columns={'Prompts':'caption'}, inplace=True)

    # Drop the 'Category' column
    drawbench_dataset_df.drop(columns='Category', inplace=True)

    # Add a column to the dataframe with the name 'source' and fill it with 'drawbench'
    drawbench_dataset_df['source'] = 'drawbench'

    # Add a new column to the dataframe with the name 'caption_id' and fill it with the index + 10000
    drawbench_dataset_df['caption_id'] = drawbench_dataset_df.index

    # Based on the splits dataframe, I want you to select the indexes where the split column is equal to 'train'
    train_indices = splits[splits['split'] == 'train']['index']

    # Same for val and test
    val_indices = splits[splits['split'] == 'val']['index']
    test_indices = splits[splits['split'] == 'test']['index']

    # Based on the train_indices, I want you to select the rows from the drawbench dataframe
    # and store it in a new dataframe called train_drawbench
    train_drawbench = drawbench_dataset_df.loc[train_indices]
    val_drawbench = drawbench_dataset_df.loc[val_indices]
    test_drawbench = drawbench_dataset_df.loc[test_indices]

    # Reindex the dataframes
    train_drawbench.reset_index(drop=True, inplace=True)
    val_drawbench.reset_index(drop=True, inplace=True)
    test_drawbench.reset_index(drop=True, inplace=True)

    # Joint Data

    # Concatenate the train_mscoco and train_drawbench dataframes and store it in train
    train = pd.concat([train_mscoco, train_drawbench])

    # Concatenate the val_mscoco and val_drawbench dataframes and store it in val
    val = pd.concat([val_mscoco, val_drawbench])

    # Concatenate the test_mscoco and test_drawbench dataframes and store it in test
    test = pd.concat([test_mscoco, test_drawbench])



    # For each dataframe I want you to shuffle the rows, and 
    # save the shuffling index 

    # Reset indexes for the dataframes

    for df in [train, val, test]:
        df.reset_index(drop=True, inplace=True)


    for split_name, df in [('train', train), ('val', val), ('test', test)]:
            shuffle_file = fr'C:\Users\User\Desktop\Research\PQPP\dataset\shuffle\{split_name}_shuffle.npy'
            if os.path.exists(shuffle_file):
                shuffle_indices = np.load(shuffle_file)
            else:
                shuffle_indices = np.random.permutation(df.index)
                np.save(shuffle_file, shuffle_indices)
            print(f'Shuffling {split_name} data')
            print(shuffle_indices)
            df = df.iloc[shuffle_indices].reset_index(drop=True)
            # Update the dataframe after shuffling
            if split_name == 'train':
                train = df
            elif split_name == 'val':
                val = df
            else:
                test = df

    # Save the train, val and test dataframes to csv files

    if not(os.path.exists(fr'../generative/{task}')):
        os.makedirs(fr'../generative/{task}')

    train.to_csv(f'../generative/ground_truth/{task}/{task}_train.csv', index=False, encoding='utf-8')
    val.to_csv(f'../generative/ground_truth/{task}/{task}_val.csv', index=False, encoding='utf-8')
    test.to_csv(f'../generative/ground_truth/{task}/{task}_test.csv', index=False, encoding='utf-8')


def generate_mscoco_data_splits(mscoco_csv_file_path):
    pass

def generate_drawbench_data_splits(drawbench_csv_file_path):
    pass

# Execute the function
splits = get_or_create_drawbench_data_split(split_path, drawbench_csv_file_path)

generate_joint_data_splits(drawbench_csv_file_path, mscoco_csv_file_path, splits)
