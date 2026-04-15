import pickle
mscoco_result_path = fr"C:\Users\User\Desktop\Research\stable-prompt-pred\retrieval_models\blip2\blip2_retrieval_results.pickle"

with open(mscoco_result_path, "rb") as mscoco_result_path:
    mscoco_data = pickle.load(mscoco_result_path)


print(mscoco_data[0])