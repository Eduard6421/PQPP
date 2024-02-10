import pandas as pd
from tqdm import tqdm

from sklearn.metrics import cohen_kappa_score

pd.set_option("display.max_columns", None)

all_users_ann = pd.read_csv("../dataset/all_users_ann_new.csv")
best_captions_df = pd.read_pickle("../dataset/best_captions_df.pickle").head(10000)


def calculate_mean_for_majority_interval(group):
    # Count occurrences in each interval
    count_negative = group["processed_option"].isin([-1, 0]).sum()
    count_positive = group["processed_option"].isin([1, 2]).sum()

    # Determine the majority interval and calculate mean
    if count_negative > count_positive:
        # Majority in [-1, 0]
        return group[group["processed_option"].isin([-1, 0])]["processed_option"].mean()
    else:
        # Majority in [1, 2] or equal (defaults to this case)
        return group[group["processed_option"].isin([1, 2])]["processed_option"].mean()


def calculate_best_model(
    series,
):
    # Corrected postfix lists
    postfix_sdxl = ["_4", "_5"]
    postfix_clip = ["_7", "_8"]

    # Filters the series by checking the postfix of each index
    series_sdxl = series[
        series.index.to_series().apply(
            lambda x: any(x.endswith(pf) for pf in postfix_sdxl)
        )
    ]
    series_clip = series[
        series.index.to_series().apply(
            lambda x: any(x.endswith(pf) for pf in postfix_clip)
        )
    ]

    # Calculate the mean score for each group
    mean_sdxl = series_sdxl.mean()
    mean_clip = series_clip.mean()

    # Print which group has the higher mean score
    if mean_sdxl > mean_clip:
        print("sdxl")
    else:
        print("clip")


def extract_for_best_model(series, dataframe, best_captions_df):
    # Corrected postfix lists
    prefixes_extracted = [
        item
        for item in series.index.tolist()
        if item.endswith("_4") or item.endswith("_5")
    ]

    # Access the items with the prefixes from the series
    series_sdxl = series[series.index.isin(prefixes_extracted)]

    group_by_prefix = series_sdxl.groupby(series_sdxl.index.str[:-2]).mean()

    text_to_id = {}

    for index, row in tqdm(dataframe.iterrows()):
        text_to_id[row["pmt_content"]] = row["annot_image_id"][:-2]
    scores = []

    for index, best_caption_row in tqdm(best_captions_df.iterrows()):
        id = text_to_id[best_caption_row["best_caption"]]
        score = group_by_prefix[id]
        scores.append(score)

    best_captions_df["score"] = scores
    best_captions_df = best_captions_df[["image_id", "best_caption", "score"]]
    best_captions_df.to_csv("../dataset/gt_for_generative_best_model.csv", index=False)


def extract_for_worst_model(series, dataframe, best_captions_df):
    # Corrected postfix lists
    prefixes_extracted = [
        item
        for item in series.index.tolist()
        if item.endswith("_7") or item.endswith("_8")
    ]

    # Access the items with the prefixes from the series
    series_sdxl = series[series.index.isin(prefixes_extracted)]

    group_by_prefix = series_sdxl.groupby(series_sdxl.index.str[:-2]).mean()

    text_to_id = {}

    for index, row in tqdm(dataframe.iterrows()):
        text_to_id[row["pmt_content"]] = row["annot_image_id"][:-2]
    scores = []

    for index, best_caption_row in tqdm(best_captions_df.iterrows()):
        id = text_to_id[best_caption_row["best_caption"]]
        score = group_by_prefix[id]
        scores.append(score)

    best_captions_df["score"] = scores
    best_captions_df = best_captions_df[["image_id", "best_caption", "score"]]
    best_captions_df.to_csv("../dataset/gt_for_generative_worst_model.csv", index=False)


def extract_for_all_models(series, dataframe, best_captions_df):
    # Corrected postfix lists
    prefixes_extracted = [
        item
        for item in series.index.tolist()
        if item.endswith("_4")
        or item.endswith("_5")
        or item.endswith("_7")
        or item.endswith("_8")
    ]

    # Access the items with the prefixes from the series
    series_all = series[series.index.isin(prefixes_extracted)]

    group_by_prefix = series_all.groupby(series_all.index.str[:-2]).mean()

    text_to_id = {}

    for index, row in tqdm(dataframe.iterrows()):
        text_to_id[row["pmt_content"]] = row["annot_image_id"][:-2]
    scores = []

    for index, best_caption_row in tqdm(best_captions_df.iterrows()):
        id = text_to_id[best_caption_row["best_caption"]]
        score = group_by_prefix[id]
        scores.append(score)

    best_captions_df["score"] = scores
    best_captions_df = best_captions_df[["image_id", "best_caption", "score"]]

    best_captions_df.to_csv(
        "../dataset/gt_for_generative_all_models_new.csv", index=False
    )


def get_control_prompt_annotations(users_annotation_df, email):
    filtered_df = users_annotation_df[users_annotation_df["email"] == email]
    return filtered_df


import numpy as np


def compute_cohen_kappa(users_annotation_df, user_email, annotator_email):
    user_df = get_control_prompt_annotations(users_annotation_df, user_email)
    annotator_df = get_control_prompt_annotations(users_annotation_df, annotator_email)

    user_df_control = user_df[user_df["pmt_common_prompt"] == "t"]
    annotator_df_control = annotator_df[annotator_df["pmt_common_prompt"] == "t"]

    user_df_control = user_df_control[["img_id", "opt_id"]]
    annotator_df_control = annotator_df_control[["img_id", "opt_id"]]
    merged_df = pd.merge(user_df_control, annotator_df_control, on="img_id")

    if len(merged_df) == 0:
        # print(
        #    "No common images for user: ",
        #    user_email,
        #    " and annotator: ",
        #    annotator_email,
        # )
        return 0

    score = cohen_kappa_score(merged_df["opt_id_x"], merged_df["opt_id_y"])

    if cohen_kappa_score == np.NaN:
        return 0

    return score


def get_best_cohen_kappa(users_annotation_df, user_email, annotator_emails):
    best_kappa = -1
    best_annotator = ""
    for annotator_email in annotator_emails:
        kappa = compute_cohen_kappa(users_annotation_df, user_email, annotator_email)
        if kappa > best_kappa:
            best_kappa = kappa
            best_annotator = annotator_email

    return best_kappa, best_annotator


def filter_for_best_annotators(users_annotation_df, annotator_emails):
    filtered_df = users_annotation_df[
        users_annotation_df["email"].isin(annotator_emails)
    ]
    return filtered_df


def generate_gt_for_generative_task(df, best_captions_df):
    annotator_emails = [
        "eduard-gabriel.poesina@my.fmi.unibuc.ro",
        "adriana16costache@gmail.com",
        "macheriejoliequejaimeaussi@gmail.com",
    ]

    # Filter out rows where columns email is in emails_to_ban
    # df = df[~df["email"].isin(emails_to_ban)]

    option_mapping = {1: 0, 2: 1, 3: 2, 4: -1}

    # Apply the mapping to the 'annot_option_id' column
    df.loc[:, "processed_option"] = df["opt_id"].map(option_mapping)

    emails = df["email"].unique()

    cohen_kappa_scores = {}

    for email in emails:
        if email in annotator_emails:
            cohen_kappa_scores[email] = 1
        else:
            kappa, best_annotator = get_best_cohen_kappa(df, email, annotator_emails)
            cohen_kappa_scores[email] = kappa

    # remove from cohen_kappa_sscores keys that have a lower score than 0.4
    removed_users = [k for k, v in cohen_kappa_scores.items() if v < 0.4]
    cohen_kappa_scores = {k: v for k, v in cohen_kappa_scores.items() if v >= 0.4}
    accepted_emails = list(cohen_kappa_scores.keys())

    df = df[df["email"].isin(accepted_emails)]

    # print removed users
    print(len(removed_users))
    print(removed_users)
    print(cohen_kappa_scores)
    print(accepted_emails)
    grouped_df = df.groupby("img_id").size().reset_index(name="counts")
    less_than_3_annotations = grouped_df[grouped_df["counts"] < 3]["img_id"].tolist()
    ids_less_than_3 = df[df["img_id"].isin(less_than_3_annotations)]["pmt_id"].unique()
    if len(ids_less_than_3) > 0:
        print(len(ids_less_than_3))
        raise Exception("Not sufficent annotators for some images")

    score_df = pd.DataFrame(cohen_kappa_scores.items(), columns=["email", "score"])
    merged_df = pd.merge(df, score_df, on="email")
    sorted_df = merged_df.sort_values(by=["img_id", "score"], ascending=[True, False])

    import plotly.express as px
    import kaleido

    remap = {
        -1: "Unrealistic",
        0: "No Relevance",
        1: "Low Relevance",
        2: "High Relevance",
    }
    merged_df["renamed_option"] = merged_df["processed_option"].map(remap)

    df_filtered = merged_df[~merged_df["img_id"].str.endswith("_6")]

    print(df_filtered["img_id"])
    import plotly.io as pio

    pio.kaleido.scope.mathjax = None
    fig = px.histogram(
        df_filtered,
        x="renamed_option",
        category_orders={
            "renamed_option": [
                "High Relevance",
                "Low Relevance",
                "No Relevance",
                "Unrealistic",
            ]
        },
    )

    fig.update_layout(
        bargap=0.25,
        yaxis_title="Number of annotations",
        xaxis_title="Annotation",
        font=dict(size=20),
    )

    fig.update_xaxes(tickangle=0, automargin=True)
    fig.update_layout(width=800)

    fig.write_image("histogram.pdf")
    # print("Max number of annotations per user", max(merged_df["email"].value_counts()))
    # print("Min number of annotations per user", min(merged_df["email"].value_counts()))
    # print(merged_df.columns)

    # plot a histogram of the processed_option using plotly

    """
    # Score df


    top3_df = sorted_df.groupby("img_id").head(3)
    result = top3_df.groupby("img_id").apply(calculate_mean_for_majority_interval)

    print("Number of accepted annotators", len(accepted_emails))
    print("Number of removed annotators", len(removed_users))
    print("Total number of annotators", len(accepted_emails) + len(removed_users))

    print("Number of annotations", len(merged_df))
    print("Mean number of annotations per image", len(merged_df) / 50000)

    min_kohen_accepted = min(cohen_kappa_scores.values())
    max_kohen_accepted = max(cohen_kappa_scores.values())
    mean_kohen_accepted = sum(cohen_kappa_scores.values()) / len(cohen_kappa_scores)

    print("Min kohen accepted", min_kohen_accepted)
    print("Max kohen accepted", max_kohen_accepted)
    print("Mean kohen accepted", mean_kohen_accepted)

    print("Mean nummber of annotations per user", len(merged_df) / len(accepted_emails))
    print(
        "Median number of annotations per user",
        merged_df["email"].value_counts().median(),
    )
    print("Max number of annotations per user", max(merged_df["email"].value_counts()))
    print("Min number of annotations per user", min(merged_df["email"].value_counts()))

    print(
        "Min number of annotations per image",
        merged_df["img_id"].value_counts().min(),
    )
    print(
        "Mean number of annotations per image",
        merged_df["img_id"].value_counts().mean(),
    )
    print(
        "Median number of annotations per image",
        merged_df["img_id"].value_counts().median(),
    )
    print(
        "Max number of annotations per image",
        merged_df["img_id"].value_counts().max(),
    )


    parsed_result = result.copy()
    # Reset hte index for pased_Result
    parsed_result = parsed_result.reset_index()
    parsed_result.columns = ["image_id", "score"]
    parsed_result.to_csv("../dataset/gt_for_generative_individual_new.csv", index=False)

    # calculate_best_model(result)
    extract_for_all_models(result, top3_df, best_captions_df)
    # extract_for_best_model(result, top3_df, best_captions_df)
    # extract_for_worst_model(result, top3_df, best_captions_df)
    # extract_for_all_models_individual_scores
    """


generate_gt_for_generative_task(all_users_ann, best_captions_df)
