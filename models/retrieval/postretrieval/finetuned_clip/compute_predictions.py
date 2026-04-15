# open test_predictions from local file test_predictions.pickle
import pickle

test_predictions_path = "./unprocessed_test_predictions.pickle"
test_predictions = pickle.load(open(test_predictions_path, "rb"))


def generate_prediction(prediction_list):

    clip_p10_predictions = []
    clip_mrr_predictions = []
        
    blip2_p10_predictions = []
    blip2_mrr_predictions = []


    import numpy as np
    print(prediction_list)
    print(np.min(prediction_list), np.mean(prediction_list), np.quantile(prediction_list, 0.5),np.max(prediction_list))
 

    for i in range(0, len(prediction_list), 50):

        limit = np.quantile(prediction_list, .5)


        clip_current_mrr = 0

        for j in range(i, i+25):
            if(prediction_list[j] >= limit):
                clip_current_mrr = 1 / (j-i+1)
                break
                
        clip_match_count = 0
        for j in range(i, i+10):
            if(prediction_list[j] >= limit):
                clip_match_count +=1
        clip_match_count /= 10


        blip2_current_mrr = 0
        blip2_match_count = 0


        for j in range(i+25, i+50):
            if(prediction_list[j] >= limit):
                blip2_current_mrr = 1 / (j-i+1)
                break

        for j in range(i+25, i+35):
            if(prediction_list[j] >= limit):
                blip2_match_count +=1

        blip2_match_count /= 10

        clip_p10_predictions.append(clip_match_count)
        clip_mrr_predictions.append(clip_current_mrr)

        blip2_p10_predictions.append(blip2_match_count)
        blip2_mrr_predictions.append(blip2_current_mrr)

    return  clip_p10_predictions, clip_mrr_predictions, blip2_p10_predictions, blip2_mrr_predictions


clip_p10_predictions, clip_mrr_predictions, blip2_p10_predictions, blip2_mrr_predictions = generate_prediction(test_predictions)

# Save the predictions to a pickle file

with open("crossmodel_drawbench_clip_p10_predictions.pickle", "wb") as file:
    pickle.dump(clip_p10_predictions, file)

with open("crossmodel_drawbench_clip_mrr_predictions.pickle", "wb") as file:
    pickle.dump(clip_mrr_predictions, file)	

with open("crossmodel_drawbench_blip2_p10_predictions.pickle", "wb") as file:
    pickle.dump(blip2_p10_predictions, file)	

with open("crossmodel_drawbench_blip2_mrr_predictions.pickle", "wb") as file:
    pickle.dump(blip2_mrr_predictions, file)