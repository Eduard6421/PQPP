# PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction (Official Repository) <a name="pqpp"></a>
This repository hosts the implementation and dataset for the scientific paper "PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction". We present our aggregation of over 1.5M prompt / query annotations obtained from over 270 annotators.

## Table of contents <a name = "table-of-contents"></a>
- [PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](#pqpp)
  - [Citation ](#citation-)
  - [About ](#about-)
  - [Note](#note)  
  - [Dataset Description](#dataset-description)
  - [Project Structure](#project-structure)
  - [Getting Started ](#getting-started)
    - [Installing Prerequisites ](#prereqs)
  - [Usage ](#usage-)
    - [Retrieval Models ](#retrieval-models)
    - [Prediction Models ](#prediction-models) 
  - [Benchmark](#benchmark)
    - [Retrieval](#benchmark-ret) 
    - [Generative](#benchmark-gen) 

  - [⛏️ Developed with ](#️developed-with)
  - [Citation ](#citation-)
  - [🎉 Acknowledgements ](#acknowledgement)
  - [License](#license)

## Citation <a name = "citation"></a>

Please cite our work if you use any material released in this repository.
```
@article{Poesina-Arxiv-2024,
      title="{PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction}", 
      author={Eduard Poesina and Adriana Valentina Costache and Adrian-Gabriel Chifu and Josiane Mothe and Radu Tudor Ionescu},
      year={2024},
      journal={arXiv preprint arXiv:2406.04746},
  }
```

## About <a name = "about"></a>
This repository hosts the annotated dataset and the implementations of the prediction models described in the original paper.
This file also hosts the extended benchmark, including models which did not pass a minimum correlation threshold of 0.1.

We add instructions and full code in order to allow other researchers to easily replicate our research or validate our approach. You can find instructions for installation and training models.
We provide ground truth data for both generative and retrieval formats in order to facilitate integrations within your own code.

## Note <a name="note"></a>
If you are interested to conduct your own research on the dataset (or any of the retrieval/generative setting models) you you can download the generated images and original MS COCO used images at:

https://fmiunibuc-my.sharepoint.com/:u:/g/personal/radu_ionescu_fmi_unibuc_ro/Eb0peYyLDVRNn0EPeY7ZwKUBAd4Yt-Zs_PtEpc-DmQ0P4A?e=oIflTJ.

# Dataset Card
## PQPP
This dataset is used to train,validate and test query performance prediction ( BLIP2 / CLIP ) and prompt performance prediction models (SDXL / GLIDE). The dataset aggregates ~1.5M annotation collected from more than 200 annotators as per our methodology described in the original paper.

#### Dataset Link
Dataset Links: 
Official repo: https://github.com/Eduard6421/PQPP
Additional Resources: https://fmiunibuc-my.sharepoint.com/personal/radu_ionescu_fmi_unibuc_ro/_layouts/15/onedrive.aspx?ga=1


#### Data Card Author(s)
- **Poesina Eduard-Gabriel, University of Bucharest** (Author) (eduardgabriel.poe@gmail.com)
- **Adriana-Valentina Costache, University of Bucharest** (Author) (adriana.costache@s.unibuc.ro)
- **Adrian-Gabriel Chifu, Aix-Marseille Universite** (Author)
- **Josiane Mothe, Aix-Marseille Universite** (Author)
- **Radu TUdor Ionescu, University of Bucharest Namee** (Author)


#### Industry Type(s)
- Academic - Tech

#### Contact Detail(s)
- **Contact:** Eduard-Gabriel Poesina, eduardgabriel.poe@gmail.com


## Dataset Overview
#### Data Subject(s)
- Non-Sensitive Data about people ( Contains Original Images from MS COCO)

#### Dataset Snapshot
Category | Data
--- | ---
Size of Dataset | 123456 MB
Number of Instances | 10000
Human Labels Collected | 1,489,836


#### Content Description
```
id - number, id of the query in MS COCO
image_id - number, id of original image of th e
best_caption - string, text containing selected prompt
blip2_rr - float, reciprocal rank for query using blip2 retrieval method
clip_rr - float, reciprocal rank for query using clip retrieval method
retrieval_avg_rr - float, average of the reciprocal rank scores of both retrieval models
blip2_pk - float, precision @ 10 for the query using blip2 retrieval method
clip_pk - float, precision @ 10 for the query using clip retrieval method
retrieval_avg_pk - float, average of the precision @ 10 scores of both retrieval methods
glide_score - human annotated generative score for the glide model 
sdxl_score - human annotated generative score for the sdxl model
avg_generative_score - average of the humman annotated generative scores
```

#### Typical Data Point
```
id,image_id,best_caption,blip2_rr,clip_rr,retrieval_avg_rr,blip2_pk,clip_pk,retrieval_avg_pk,glide_score,sdxl_score,avg_generative_score
319365,363951,Black and white of windsurfers on a lake.,1.0,1.0,1.0,0.1,0.1,0.1,0.5,2.0,1.25
```



## Dataset Folder Description <a name="dataset-description"></a>
Our dataset consists of two major parts:

        1. train.csv/validation.csv/test.csv (containing MS COCO image id, p@10 /RR scores for retrieval setting, and the scores for the generative setting )
        2. image folder ( contains the SDXL/GLIDE generated images alongside the original MS COCO image).
    
The image folder has the following structure:

        images\
            {IMG_ID_1}\
                \image_4.png - Image Generated by SDXL 
                \image_5.png - Image Generated by SDXL 
                \image_6.png - Ground Truth, Original MS-COCO Image
                \image_7.png - Image Generated by GLIDE
                \image_8.png - Image Generated by GLIDE
            {IMG_ID_2}\
            .
            .
            .        
            {IMG_ID_N}\

The suffies _4, _5 denote generation by SDXL.

THe suffix _6 denote the MS-COCO dataset.

The suffixes _7,_8 denote generation by GLIDE.

## Project structure <a name="project-structure"></a>

    The project is structured as following:

    \PQPP
        \dataset - folder containing the annotated dataset
            \generative_annotation_scores
                \ gt_for_generative_all_models.csv - Score for each query in the generative seetting as described in the paper.
                \ gt_for_generative_glide_new.csv - Score for each query of the glide method.
                \ gt_for_generative_all_models.csv - Score for each query of SDXL method.
            \retrieval_model_scores
                \ avg_scores_rr.pickle - the average rr score per query of the two retrieval models.
                \ avg_scores_p10.pickle - the average p@10 score per query  of the two retrieval models.
                \ blip_2_rr_scores_map.pickle - the rr score per query of the BLIP2 model.
                \ blip_2_pk_scores_map.pickle - the P@10 score per query of the BLIP2 model.
                \ clip_rr_scores.pickle - the rr score per query for the CLIP model.
                \ clip_pk_scores.pickle - the p@10 score per query for the CLIP model.
            \ all_users_ann_new.csv - Original generative setting annotations. Anonymized file
            \ best_captions_df.pickle - File containing extra information about each query caption


            === Files to be user for training new models of replicating the current CSV files include query caption, image ids, scores for the retrieval and generative settings === 


            \ dataset.csv - Centralized ground truth file which contains all data.
            \ train.csv - File containing training data split. (60% of data)
            \ vaidation.csv - Centralized ground truth file for validation data. (20% of data)
            \ test.csv - Centralized ground truth file for test data.  (20% of data)

        \pipelines - folder containing scripts to generate images for the generative setting
        \predictors - folder containing performance predictors as described in the paper
            \ correlation_cnn - Contains the CNN-based approach inspired by Sun. et al
            \ finetuned_bert - Contains the finetuned bert model training script
            \ lingvistic_features - Contains the lingvistic features model training script
            \ finetuned_clip - Contains the finetuned clip model trainig script described in our research
            \ neural_embeddings - Contains the implementation of Arabzadeh et al.
            \ query_drift - Contains script to implement query drift.
            \ score-varation - Contains the score-variation deadline
        \retrieval_model_annotations - folder containing scripts to kickstart annotation process for retrieval and train automatic retrieval groundtruth computation
        \retrieval_models - folder containg scripts to perform text-to-image- search
    
## Getting started <a name="getting-started"></a>

### Installing Pre-requisites <a name="prereqs"></a>
In order to run our models you will need to install the requirements found in requirements.txt

    pip install -r requirements.txt

## Usage <a name="usage"></a>

### Retrieval Models <a name="retrieval-models"></a>

The retrieval models can be found at:

    \retrieval_models
        \clip
            \clip_retrieval.py - CLIP Retrieval Run. Saves Image ids returned.
            \save_clip_scores.py - CLIP Retrieval Run. Saves the scores for the retrieval.
            \generate_clip_embeddings.py - Generates CLIP embeddings for the MS-COCO dataset
            \generate_clip_query_embeddings.py - Generates CLIP embeddings for queries

        \blip2
            \blip2_retrieval.py - BLIP-2 Retrieval Run. Saves Image ids returned.
            \save_blip2_scores.py - BLIP-2 Retrieval Run. Saves the scores for the retrieval.
            \generate_blip2_embeddings.py - Generates BLIP-2 embeddings for the MS-COCO image dataset
            \generate_blip2_query_embeddings.py - Generates BLIP-2 embeddings for queries

        \compute_scores.py - Allow the computation of P@10 and RR scores.
        \generate_average_scores.py - Computes the average of the computed scores (CLIP/BLIP-2)

### Predictions models <a name="prediction-models"></a>

Prediction models can be found at:

    \predictors
        \correlation_cnn
        \finetuned_clip
        \neural_embeddings
        \score-variation


## Instructions

Dataset Research 
1. Clone the github from the official repo
2. Download the support files if 


Paper Reproduction

Retrieval-Models Results Generation:
1. Download the extra resource containing the ![processed retrieval groundtruth]()
2. If you wish to reproduce the retrieval models go to the retrieval models folder and select the preferred model.
3. Run ```python {model}_retrieval.py```. This will allow you to rerun the selected model retrieval process.
4. Run ```python save_{model}_scores.py```. This will compute the scores of the retrieval model and save it to 



#### Domain(s) of Application
Machine Learning,Computer Vision, Query Performance Prediction, Prompt Performance Prediction, Retrieval Models, Generative Models


## Provenance
### Collection
#### Method(s) Used
- Crowdsourced - Volunteer


### Dataset Version and Maintenance
#### Maintenance Status
**Actively Maintained** - No new versions will be made
available, but this dataset will
be actively maintained,
including but not limited to
updates to the data.


#### Version Details
**Current Version:** 1.0
**Last Updated:** 05/2024
**Release Date:** 05/2024



## Complete benchmark <a name="benchmark"></a>



Predictors

1. Go to the selected predictor  folder ( ex: predictors/correlation_cnn)
2. Select the setting you wish to run ( retrieval / generative) and enter the appropiate folder.
3. For trainable models you will find a script entitled gen

### Retrieval <a name="benchmark-ret"> </a>

![Retrieval Results Placeholder](retrieval_results.png)

### Generative <a name="benchmark-gen"> </a>

![Generative Results Placeholder](generative_results.png)

## Developed with <a name="developed-with"></a>

Annotation platform was developed with the following technologies:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Django](https://img.shields.io/badge/django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Context-API](https://img.shields.io/badge/Context--Api-000000?style=for-the-badge&logo=react)

Research was conducted using the following technologies:
    
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

OpenAI CLIP Model
https://github.com/openai/CLIP

OpenAI GLIDE Model
https://github.com/openai/glide-text2im

Salesforce BLIP-2 Model
https://github.com/salesforce/LAVIS/tree/main/projects/blip2

StabilityAI SDXL Model
https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
We thank all researchers for their implication and expertise and annotators for their incredible work amounting to our dataset.


## License <a name = "license"></a>
The MS COCO annotations are released under a Creative Commons Attribution 4.0 License https://cocodataset.org/#termsofuse.
THe MS COCO images are subject to Flickr Terms of Use https://www.flickr.com/creativecommons/

We release our annotations and generated images maintaining the Creative Commons Attribution 4.0 License https://cocodataset.org/#termsofuse
[CC BY 4.0]([https://choosealicense.com/licenses/mit/](https://creativecommons.org/licenses/by/4.0/deed))
