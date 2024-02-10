import pickle


blip2_p10_score = "./blip2/pk_scores_map_new.pickle"
clip_p10_score = "./clip/pk_scores_map_new.pickle"
blip2_p10 = pickle.load(open(blip2_p10_score, "rb"))
clip_p10 = pickle.load(open(clip_p10_score, "rb"))


avg_score = {}
for key in blip2_p10:
    blip2_score = blip2_p10[key]
    clip_score = clip_p10[key]
    avg_score[key] = (blip2_score + clip_score) / 2


pickle.dump(avg_score, open("./avg_scores_p10_new.pickle", "wb"))


blip2_mrr_score = "./blip2/mrr_scores_map_new.pickle"
clip_mrr_score = "./clip/mrr_scores_map_new.pickle"
blip2_mrr = pickle.load(open(blip2_mrr_score, "rb"))
clip_mrr = pickle.load(open(clip_mrr_score, "rb"))


avg_score = {}
for key in blip2_mrr:
    blip2_score = blip2_mrr[key]
    clip_score = clip_mrr[key]
    avg_score[key] = (blip2_score + clip_score) / 2


pickle.dump(avg_score, open("./avg_scores_mrr_new.pickle", "wb"))
