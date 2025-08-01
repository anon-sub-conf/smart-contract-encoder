from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from sentence_transformers import util
from smart_contract_encoder.load_data import *


def create_translation_pairs_dataset(field):
    df = load_dataset(file_type="merged", split="train")
    print("Creating translation pairs dataset from the processed dataset (train split)")
    train_dataset = pd.DataFrame()
    train_dataset["anchor"] = df[field]
    train_dataset["positive"] = df["func_documentation"]
    train_dataset = train_dataset[['anchor','positive']]
    records = train_dataset.to_dict(orient="records")
    eval_size = 2000
    train_size = len(records) - eval_size
    data_records = records[:train_size]
    eval_records = records[train_size:]
    output_json = {
        "version": "0.0.0",
        "data": data_records,
        "eval": eval_records
    }
    print(train_dataset)
    save_dataset(dataset=output_json,file_type="training",field=field, split="train", training_dataset_type="translation_pairs")


def get_dist_matrx(embeddings):
    return util.cos_sim(embeddings,embeddings)


def comp_pair_combinations(dataset):
    combinations = set()
    for _, row in dataset.iterrows():
        anchor = row.anchor
        positives = row.positive
        for pos in positives:
            combinations.add((anchor, pos))
    return combinations


def get_pair_combinations(dist_matrx, df, pos_threshold):
    data = set()
    dist_matrx = dist_matrx.numpy()
    visited = set()
    for i, matrx in tqdm(enumerate(dist_matrx)):
        anchor = df.loc[i]
        if i in visited:
            continue
        anchor_addr = anchor['contract_address']
        # anchor_doc = anchor['func_documentation']
        anchor_code = anchor['code']
        anchor_idx = i
        positives = []
        for j, dist in enumerate(matrx):
            if i == j:
                continue
            elif dist >= pos_threshold:
                candidate = df.loc[j]
                if anchor_addr != candidate['contract_address'] and \
                anchor_code != candidate['code']:
                    positives.append((dist, j))
                    visited.add(j)
        if len(positives) > 0:
            # positives = [i for (_,i) in sorted(positives, reverse=True)][:100]
            positives = [i for (_,i) in sorted(positives, reverse=True)]
            for p in positives:
                visited.add(p)
            data.add((anchor_idx, tuple(positives)))
    dataset = pd.DataFrame(data=data, columns=["anchor","positive"])
    return dataset

def get_pair_code(dataset, combinations, field):
    data = []
    for (a, p) in combinations:
        data.append((dataset.loc[a][field], dataset.loc[p][field]))
    train_dataset = pd.DataFrame(data=data, columns=["anchor","positive"])
    return train_dataset


def create_pairs_dataset(split, field):
    df = load_dataset(file_type="merged", split=split)
    print(f"Creating pairs dataset from the merged dataset (train split) on the field {field}")
    baseline_enc = "sentence_encoder"
    baseline_enc_version = "untrained"
    baseline_field = "func_documentation"
    baseline_embeddings = load_dataset(file_type="embeddings", split=split, encoder=baseline_enc, encoder_version=baseline_enc_version, field=baseline_field)
    baseline = np.stack(baseline_embeddings['embeddings'].to_list())
    dist_matrx = get_dist_matrx(baseline)
    train_data = get_pair_combinations(dist_matrx, df, 0.72)
    combinations = comp_pair_combinations(train_data)
    train_dataset = get_pair_code(df, combinations, field)
    train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)
    print(train_dataset)
    train_dataset = train_dataset[['anchor','positive']]
    records = train_dataset.to_dict(orient="records")
    eval_size = 4000
    train_size = len(records) - eval_size
    data_records = records[:train_size]
    eval_records = records[train_size:]
    output_json = {
        "version": "0.0.0",
        "data": data_records,
        "eval": eval_records
    }
    save_dataset(dataset=output_json,file_type="training", field=field, split="train", training_dataset_type="positive_pairs")



