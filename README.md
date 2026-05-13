# PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction (Official Repository) - CVPR 2025<a name="pqpp"></a>
This repository contains the implementation and dataset for the paper "PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction", accepted at CVPR 2025. We have compiled over 1.5 million prompt and query annotations from more than 270 annotators.

The dataset includes human annotations evaluating image retrieval performance using BLIP-2 and CLIP, as well as prompt performance in generative models such as GLIDE and SDXL. In addition to the core MS-COCO-based split, we release a DrawBench extension for out-of-distribution evaluation.

## Table of contents <a name="table-of-contents"></a>
- [PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](#pqpp)
  - [Citation](#citation)
  - [About](#about)
  - [Note](#note)
  - [Data Card](#data-card)
    - [Dataset Overview](#dataset-overview)
    - [Dataset Folder Description](#dataset-description)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [Installing Pre-requisites](#prereqs)
    - [Loading Data](#loading-data)
  - [Usage](#usage)
    - [Retrieval Process](#retrieval-process)
    - [Prediction Models](#prediction-models)
    - [Ground-Truth & Correlation Utilities](#utilities)
  - [Complete Benchmark](#benchmark)
    - [Retrieval](#benchmark-ret)
    - [Generative](#benchmark-gen)
  - [Developed with](#developed-with)
  - [Acknowledgements](#acknowledgement)
  - [License](#license)

## Citation <a name="citation"></a>
Please cite our work if you use any material released in this repository.

1. Eduard Poesina, Adriana Valentina Costache, Adrian-Gabriel Chifu, Josiane Mothe, Radu Tudor Ionescu. PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction. In Proceedings of CVPR, pp. 28651-28661, 2025.

Bibtex entry:
```
@inproceedings{Poesina-CVPR-2025,
  title="{PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction}",
  author={Poesina, Eduard and Costache, Adriana Valentina and Chifu, Adrian-Gabriel and Mothe, Josiane and Ionescu, Radu Tudor},
  booktitle={Proceedings of CVPR},
  pages={28651--28661},
  year={2025}
}
```

## About <a name="about"></a>
This repository hosts the annotated dataset and the implementations of the prediction models described in the original paper. It also hosts the extended benchmark, including models which did not pass the minimum correlation threshold of 0.1 reported in the paper.

We provide installation instructions, training scripts, and the per-split ground-truth files for both the generative and retrieval settings so that other researchers can easily replicate our results or integrate PQPP into their own code.

## Note <a name="note"></a>
If you want to conduct your own research on the dataset (or any of the retrieval/generative setting models), you can download the generated images and the original MS-COCO images we used from the support bundle linked under [Getting Started](#getting-started).

Direct download for the SDXL / GLIDE generated image archive: [images.zip]([https://fmiunibuc-my.sharepoint.com/personal/radu_ionescu_fmi_unibuc_ro/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fradu%5Fionescu%5Ffmi%5Funibuc%5Fro%2FDocuments%2FPQPP%2Fimages%2Ezip](https://fmiunibuc-my.sharepoint.com/:u:/g/personal/radu_ionescu_fmi_unibuc_ro/Eb0peYyLDVRNn0EPeY7ZwKUBAd4Yt-Zs_PtEpc-DmQ0P4A?e=oIflTJ)).

## Data Card <a name="data-card"></a>

### Dataset Overview <a name="dataset-overview"></a>
#### Data Subject(s)
- Non-Sensitive Data about people (contains original images from MS-COCO).

#### Dataset Snapshot
Category | Data
--- | ---
Size of Dataset | 34 GB
Number of Instances | 10,000 (MS-COCO split) + 200 (DrawBench extension)
Human Labels Collected | 1,489,836

#### Content Description

Generative ground-truth CSVs (`dataset/generative/ground_truth/**`):
```
caption_id - number, id of the query in MS COCO
caption    - string, prompt text
score      - float, human-annotated generative score
             (average across annotators for the given generator)
source     - string, origin of the prompt (e.g. "mscoco")
```

Retrieval ground-truth CSVs (`dataset/retrieval/ground_truth/**`):
```
prompt          - string, query text
source          - string, origin of the prompt (e.g. "mscoco")
index           - number, sequential index in the split
caption_id      - number, id of the query in MS COCO
precision       - float, precision @ 10 for the query
reciprocal_rank - float, reciprocal rank for the query
```
For the per-model folders (`blip2/`, `clip/`) the scores are those of the named retrieval model; for `average/` the scores are the mean of BLIP-2 and CLIP.

Retrieval relevance (`dataset/retrieval/ground_truth/retrieval_{train,val,test}_gt.pickle`) stores, for each query, the list of MS-COCO image ids considered manually relevant.

#### Typical Data Point (generative, average over models)
```
caption_id,caption,score,source
11042,A large slice of cheese pizza on a paper plate.,1.5,mscoco
```

### Dataset Folder Description <a name="dataset-description"></a>
The dataset is split into three 60/20/20 partitions (train/val/test) shared across the generative and retrieval tasks. For each task we publish per-model ground truth and an averaged ground truth, plus a DrawBench out-of-distribution extension.

```
dataset/
├── generative/
│   ├── ground_truth/
│   │   ├── average/{train,val,test}.csv     # mean generative score (GLIDE + SDXL)
│   │   ├── glide/{train,val,test}.csv       # GLIDE-only scores
│   │   └── sdxl/{train,val,test}.csv        # SDXL-only scores
│   ├── drawbench/                           # DrawBench OOD extension
│   │   ├── drawbench_generative_task_average_score_gt.csv
│   │   ├── drawbench_generative_task_glide_score_gt.csv
│   │   ├── drawbench_generative_task_sdxl_score_gt.csv
│   │   └── drawbench_split.csv
│   └── mscoco/                              # full-dataset (non-split) generative GTs
│       ├── mscoco_generative_task_average_score_gt.csv
│       ├── mscoco_generative_task_glide_score_gt.csv
│       └── mscoco_generative_task_sdxl_score_gt.csv
├── retrieval/
│   └── ground_truth/
│       ├── average/{train,val,test}.csv     # averaged BLIP-2 + CLIP P@10 / RR
│       ├── blip2/blip2_retrieval_{train,val,test}_results.csv
│       ├── clip/clip_retrieval_{train,val,test}_results.csv
│       └── retrieval_{train,val,test}_gt.pickle   # relevant image ids per query
├── shuffle/{train,val,test}_shuffle.npy            # deterministic split indices
├── dataset_processing/alter_dataset_contents.py    # split/regeneration helper
└── drawbench_annotation.csv                        # raw DrawBench prompts + scores
```

The image folder is distributed in the support bundle (see [Getting Started](#getting-started)) and follows the structure below:
```
images/
    {IMG_ID_1}/
        image_4.png  # SDXL generation
        image_5.png  # SDXL generation
        image_6.png  # Original MS-COCO image (ground truth)
        image_7.png  # GLIDE generation
        image_8.png  # GLIDE generation
    {IMG_ID_2}/
    ...
```

## Project Structure <a name="project-structure"></a>
```
PQPP/
├── dataset/                    # ground-truth annotations + splits (see above)
├── pipelines/                  # image-generation pipelines used to produce the dataset images
│   ├── pipeline_start.py               # entry point that dispatches to the chosen generator
│   ├── glide_pipeline.py               # OpenAI GLIDE generation
│   ├── stable_difussion_xl_base.py     # Stable Diffusion XL generation (SDXL)
│   ├── stable_difussion_2_pipeline.py  # Stable Diffusion 2 generation
│   └── stable_difussion_2_1_pipeline*.py  # SD 2.1 variants (kept for reproducibility)
├── retrieval_process/          # text-to-image retrieval (BLIP-2, CLIP)
│   ├── blip2/
│   │   ├── blip2_retrieval.py                       # run BLIP-2 retrieval
│   │   ├── blip2_full_retrieval_results.py          # aggregate full-set results
│   │   ├── generate_blip2_query_embeddings.py       # query-side embeddings
│   │   ├── generate_blip2_drawbench_embeddings.py   # DrawBench query embeddings
│   │   ├── blip2_query_embeddings/                  # cached query embeddings (pkl)
│   │   └── retrieval_{train,val,test}_scores.pickle
│   └── clip/
│       ├── clip_retrieval.py                        # run CLIP retrieval
│       ├── clip_retrieval_merge.py                  # merge per-shard results
│       ├── clip_full_retrieval_results.py
│       ├── compute_clip_retrieval_score.py          # P@10 / RR computation
│       ├── generate_clip_query_embeddings.py
│       ├── generate_clip_drawbench_embeddings.py
│       └── clip_query_embeddings/                   # cached query embeddings (pkl)
├── models/                     # performance-prediction models (code only)
│   ├── generative/postgenerative/
│   │   ├── correlation_cnn/    # CNN-based predictor (Sun et al.-inspired)
│   │   └── finetuned_clip/     # fine-tuned CLIP predictor (our approach)
│   └── retrieval/
│       ├── preretrieval/       # pre-retrieval predictors (neural QPP)
│       └── postretrieval/
│           ├── correlation_cnn/
│           └── finetuned_clip/
├── compute_drawbench_gt.py         # build DrawBench ground-truth CSVs
├── compute_gt_correlations.py      # inter-model / inter-split correlations
├── generate_avg_retrieval_scrores.py  # produce averaged retrieval scores
├── requirements.txt
└── Annotation methodology - Retrieval.pdf
```

### What lives in `models/`
Each `correlation_cnn/` and `finetuned_clip/` subfolder contains the same canonical set of scripts, adapted to the task:
- `model.py` — network definition.
- `generate_dataset.py` — converts the CSV/pickle ground truth into tensors for training.
- `same_dataset.py` / `cross_dataset.py` — train / eval loops (in-domain vs. cross-domain).
- `compute_predictions.py` — run inference and dump per-query predictions.
- `compute_correlations.py`, `compute_samedataset_correlations.py`, `compute_crossdataset_correlations.py`, `compute_crosstask_correlations.py` — Pearson / Spearman / Kendall evaluations against the ground truth.
- `make_figure.py` — t-SNE / calibration plots reported in the paper.

### What lives in `retrieval_process/`
BLIP-2 and CLIP each expose the same pipeline: first generate query embeddings, then run retrieval against a pre-built image index, then compute P@10 and RR. Cached query embeddings for the three MS-COCO splits are included so retrieval can be re-scored without re-encoding the text corpus.

## Getting Started <a name="getting-started"></a>

### Instructions
1. Clone this repository.
2. Download the support bundles (generated images) from:
   - [SharePoint 1 - Images generated for MS COCO prompts](https://fmiunibuc-my.sharepoint.com/:u:/g/personal/radu_ionescu_fmi_unibuc_ro/Eb0peYyLDVRNn0EPeY7ZwKUBAd4Yt-Zs_PtEpc-DmQ0P4A?e=oIflTJ)
   - [SharePoint 2 - Images generated for DrawBench prompts](https://fmiunibuc-my.sharepoint.com/:u:/g/personal/radu_ionescu_fmi_unibuc_ro/IQCmS22QmoruTJUmoTvOy8K1ASTcGUJ7Fnup_uqQGo0vZT4?e=9gGEkX)
4. Unarchive the bundle inside the repository root so that an `images/` directory appears next to `dataset/`.

### Installing Pre-requisites <a name="prereqs"></a>
```
pip install -r requirements.txt
```

### Loading Data <a name="loading-data"></a>
```python
import pandas as pd

# Generative task: averaged GLIDE + SDXL scores, test split
gen = pd.read_csv('./dataset/generative/ground_truth/average/average_test.csv')

# Retrieval task: averaged BLIP-2 + CLIP P@10 / RR, test split
ret = pd.read_csv('./dataset/retrieval/ground_truth/average/average_test.csv')

# Map image paths from the support bundle (requires images/ unpacked at repo root)
gen['sdxl1']    = gen['caption_id'].apply(lambda x: f'images/{x}/image_4.png')
gen['sdxl2']    = gen['caption_id'].apply(lambda x: f'images/{x}/image_5.png')
gen['gt_image'] = gen['caption_id'].apply(lambda x: f'images/{x}/image_6.png')
gen['glide1']   = gen['caption_id'].apply(lambda x: f'images/{x}/image_7.png')
gen['glide2']   = gen['caption_id'].apply(lambda x: f'images/{x}/image_8.png')

print(gen.iloc[0])
```

## Usage <a name="usage"></a>

### Retrieval Process <a name="retrieval-process"></a>
To reproduce the retrieval scores used as ground truth for the retrieval task:
1. (Optional) Regenerate query embeddings: `python retrieval_process/{blip2,clip}/generate_{blip2,clip}_query_embeddings.py` — cached embeddings are already provided for the three splits.
2. Run retrieval against the MS-COCO image index: `python retrieval_process/{blip2,clip}/{blip2,clip}_retrieval.py`.
3. Compute P@10 and RR: `python retrieval_process/clip/compute_clip_retrieval_score.py` (same idea for BLIP-2 via `blip2_full_retrieval_results.py`).
4. Average the two retrieval models: `python generate_avg_retrieval_scrores.py`.

For the DrawBench extension, use `generate_{blip2,clip}_drawbench_embeddings.py` followed by `compute_drawbench_gt.py`.

### Prediction Models <a name="prediction-models"></a>
Each predictor is self-contained. A typical in-domain run looks like:
```
cd models/generative/postgenerative/finetuned_clip
python generate_dataset.py          # materialize training tensors
python same_dataset.py              # train + eval in-domain
python compute_predictions.py       # dump per-query predictions
python compute_samedataset_correlations.py
```
Cross-task (generative → retrieval, etc.) and cross-dataset (MS-COCO → DrawBench) evaluations are exposed via the `cross_dataset.py` / `compute_crosstask_correlations.py` / `compute_crossdataset_correlations.py` scripts in the same folder.

Pre-retrieval predictors live under `models/retrieval/preretrieval/` (`neural_qpp.py`, `compute_correlations.py`).

### Ground-Truth & Correlation Utilities <a name="utilities"></a>
- `compute_gt_correlations.py` — correlations between per-model and averaged ground truths, across splits and tasks (used to build the tables in the paper).
- `compute_drawbench_gt.py` — produces the DrawBench generative-task ground-truth CSVs from the raw annotations.
- `generate_avg_retrieval_scrores.py` — averages BLIP-2 and CLIP retrieval scores to produce `dataset/retrieval/ground_truth/average/*.csv`.

#### Domain(s) of Application
Machine Learning, Computer Vision, Query Performance Prediction, Prompt Performance Prediction, Retrieval Models, Generative Models.

## Provenance
### Collection
#### Method(s) Used
- Crowdsourced — Volunteer.

### Dataset Version and Maintenance
#### Maintenance Status
**Actively Maintained** — No new versions are planned, but the dataset will continue to receive corrections and minor updates.

#### Version Details
**Current Version:** 1.1 (CVPR 2025 camera-ready; adds DrawBench extension and per-model ground-truth CSVs)
**Last Updated:** 04/2025
**Release Date:** 05/2024

## Complete Benchmark <a name="benchmark"></a>

### Retrieval <a name="benchmark-ret"></a>
See the retrieval tables in the paper.

### Generative <a name="benchmark-gen"></a>
See the generative tables in the paper.

## Developed with <a name="developed-with"></a>
Annotation platform:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
![Django](https://img.shields.io/badge/django-%23092E20.svg?style=for-the-badge&logo=django&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Context-API](https://img.shields.io/badge/Context--Api-000000?style=for-the-badge&logo=react)

Research stack:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

OpenAI CLIP — https://github.com/openai/CLIP
OpenAI GLIDE — https://github.com/openai/glide-text2im
Salesforce BLIP-2 — https://github.com/salesforce/LAVIS/tree/main/projects/blip2
StabilityAI SDXL — https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

## Acknowledgements <a name="acknowledgement"></a>
We thank all researchers for their involvement and expertise, and all the annotators for their incredible work, which is the foundation of this dataset.

## License <a name="license"></a>
The MS-COCO annotations are released under a Creative Commons Attribution 4.0 License — https://cocodataset.org/#termsofuse.
The MS-COCO images are subject to the Flickr Terms of Use — https://www.flickr.com/creativecommons/.

We release our annotations and generated images under the Creative Commons Attribution 4.0 License — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed).
