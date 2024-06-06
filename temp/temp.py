import pickle
import pandas as pd


blip2_mrr_scores_map_new = pickle.load(open("blip2_mrr_scores_map_new.pickle", "rb"))
blip2_pk_scores_map_new = pickle.load(open("blip2_pk_scores_map_new.pickle", "rb"))


clip_mrr_scores_map_new = pickle.load(open("clip_mrr_scores_map_new.pickle", "rb"))
clip_pk_scores_map_new = pickle.load(open("clip_pk_scores_map_new.pickle", "rb"))


generative_score_avg = pd.read_csv("..\dataset\gt_for_generative_all_models_new.csv")
generative_score_sdxl = pd.read_csv("..\dataset\gt_for_generative_sdxl_new.csv")
generative_score_glide = pd.read_csv("..\dataset\gt_for_generative_glide_new.csv")

best_captions = pd.read_pickle("..\dataset\\best_captions_df.pickle").head(10000)


# Join the generative scores with the best captions dataframe on generative_scores['best_caption'] == best_captions['caption']

generative_sdxl_scores = generative_score_sdxl["score"].tolist()
generative_glide_scores = generative_score_glide["score"].tolist()
generative_avg_scores = generative_score_avg["score"].tolist()


blip2_mrr_scores_map_new = [blip2_mrr_scores_map_new[i] for i in range(10000)]
blip2_pk_scores_map_new = [blip2_pk_scores_map_new[i] for i in range(10000)]
clip_mrr_scores_map_new = [clip_mrr_scores_map_new[i] for i in range(10000)]
clip_pk_scores_map_new = [clip_pk_scores_map_new[i] for i in range(10000)]

best_captions["sdxl_score"] = generative_sdxl_scores
best_captions["glide_score"] = generative_glide_scores
best_captions["avg_generative_score"] = generative_avg_scores

best_captions["blip2_mrr"] = blip2_mrr_scores_map_new
best_captions["clip_mrr"] = clip_mrr_scores_map_new
best_captions["retrieval_avg_mrr"] = (
    best_captions["blip2_mrr"] + best_captions["clip_mrr"]
) / 2

best_captions["blip2_pk"] = blip2_pk_scores_map_new
best_captions["clip_pk"] = clip_pk_scores_map_new
best_captions["retrieval_avg_pk"] = (
    best_captions["blip2_pk"] + best_captions["clip_pk"]
) / 2


best_captions[
    [
        "id",
        "image_id",
        "best_caption",
        "blip2_mrr",
        "clip_mrr",
        "retrieval_avg_mrr",
        "blip2_pk",
        "clip_pk",
        "retrieval_avg_pk",
        "glide_score",
        "sdxl_score",
        "avg_generative_score",
    ]
].to_csv("..\dataset\\dataset.csv", index=False)
