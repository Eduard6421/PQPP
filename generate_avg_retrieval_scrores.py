import pandas as pd

blip2_train_results_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\blip2\blip2_retrieval_train_results.csv"
blip2_val_results_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\blip2\blip2_retrieval_val_results.csv"
blip2_test_results_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\blip2\blip2_retrieval_test_results.csv"

clip_train_results_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\clip\clip_retrieval_train_results.csv"
clip_val_results_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\clip\clip_retrieval_val_results.csv"
clip_test_results_path = fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\clip\clip_retrieval_test_results.csv"


blip2_train_results = pd.read_csv(blip2_train_results_path)
blip2_val_results = pd.read_csv(blip2_val_results_path)
blip2_test_results = pd.read_csv(blip2_test_results_path)

clip_train_results = pd.read_csv(clip_train_results_path)
clip_val_results = pd.read_csv(clip_val_results_path)
clip_test_results = pd.read_csv(clip_test_results_path)

# select all columns from blip2, except score which should be the average of both blip2 and clip

avg_train_results = blip2_train_results.drop(columns=["precision","reciprocal_rank"])
avg_val_results = blip2_val_results.drop(columns=["precision","reciprocal_rank"])
avg_test_results = blip2_test_results.drop(columns=["precision","reciprocal_rank"])

# remove unnamed columsn from all dataframes

avg_test_results = avg_test_results.loc[:, ~avg_test_results.columns.str.contains('^Unnamed')]
avg_val_results = avg_val_results.loc[:, ~avg_val_results.columns.str.contains('^Unnamed')]
avg_train_results = avg_train_results.loc[:, ~avg_train_results.columns.str.contains('^Unnamed')]


avg_train_results["precision"] = (blip2_train_results["precision"] + clip_train_results["precision"]) / 2
avg_val_results["precision"] = (blip2_val_results["precision"] + clip_val_results["precision"]) / 2
avg_test_results["precision"] = (blip2_test_results["precision"] + clip_test_results["precision"]) / 2

avg_train_results["reciprocal_rank"] = (blip2_train_results["reciprocal_rank"] + clip_train_results["reciprocal_rank"]) / 2
avg_val_results["reciprocal_rank"] = (blip2_val_results["reciprocal_rank"] + clip_val_results["reciprocal_rank"]) / 2
avg_test_results["reciprocal_rank"] = (blip2_test_results["reciprocal_rank"] + clip_test_results["reciprocal_rank"]) / 2

avg_train_results.to_csv(fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\average\average_train.csv")
avg_val_results.to_csv(fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\average\average_val.csv")
avg_test_results.to_csv(fr"C:\Users\User\Desktop\Research\PQPP\dataset\retrieval\ground_truth\average\average_test.csv")