# PQPP <a name="pqpp"></a>
This repository hosts the implementation and dataset for the scientific paper "PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction".



## Table of contents <a name = "table-of-contents"></a>
- [PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](#pqpp)
  - [📝 Table of Contents ](#-table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
    - [Installing Prerequisites ](#installing-prerequisites-)
  - [Usage ](#usage-)
  - [⛏️ Developed with ](#️-developed-with-)
  - [Citation ](#citation-)
  - [🎉 Acknowledgements ](#acknowledgement)



## About <a name = "about"></a>
This repository hosts the annotated dataset and the implementations of the prediction models described in the original paper.
This file also hosts the extended benchmark, including models which did not pass a minimum correlation threshold of 0.1.

    
    \PQPP
        \dataset - folder containing the annotated dataset
            \ all_users_ann_new.csv - Original generative setting annotations. Anonymized file
            \ avg_scores_mrr.pickle - Average RR for each query in the retrieval setting.
            \ avg_scores_p10.pickle - Average P@10 for each query in the retrieval setting.
            \ best_captions_df.pickle - File containing extra information about each query caption
            \ ground_truth.csv - Centralized ground truth file. This is the file to use if you plan to train new models or study the results in a "clean" method.
            \ gt_for_generative_all_models.csv - Score for each query in the generative seetting as described in the paper.
            \ merged_retrieval_gt.pickle - File containing the ground truth image matches for the retrieval setting.
            \ train_examples_new.pickle - File containing annotation for automated retrieval model training
            \ val_examples_new.pickle - File containg validation split 
            \ test_examples_new.pickle
        \pipelines - folder containing scripts to generate images for the generative setting
        \predictors - folder containing performance predictors as described in the paper
            \ correlation_cnn - Contains the CNN-based approach inspired by Sun. et al
            \ finetuned_bert - Contains the finetuned bert model training script
            \ finetuned_clip - Contains the finetuned clip model trainig script described in our research
            \ neural_embeddings - Contains the implementation of Arabzadeh et al.
            \ query_drift - Contains script to implement query drift.
            \ score-varation - Contains the score-variation deadline
        \retrieval_model_annotations - folder containing scripts to kickstart annotation process for retrieval and train automatic retrieval groundtruth computation
        \retrieval_models - folder containg scripts to perform text-to-image- search
    

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
We thank all researchers and data annotators behind the datasets and the developed baselines!

## License
[MIT](https://choosealicense.com/licenses/mit/)