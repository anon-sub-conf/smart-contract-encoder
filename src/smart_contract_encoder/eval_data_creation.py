from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import util
from smart_contract_encoder.encoder import *
from sentence_transformers.evaluation import InformationRetrievalEvaluator
# from smart_contract_encoder.ir_evaluator import InformationRetrievalEvaluator
from smart_contract_encoder.load_data import *


def create_query_dataset(split, encoder, encoder_version, field, model_to_load = None):
    df = load_dataset(file_type="merged", split=split)
    baseline_enc = "sentence_encoder"
    baseline_enc_version = "untrained"
    baseline_field = "func_documentation"
    baseline_embeddings = load_dataset(file_type="embeddings", split=split, encoder=baseline_enc, encoder_version=baseline_enc_version, field=baseline_field)
    baseline_embeddings = np.stack(baseline_embeddings['embeddings'].to_list())
    query_indices = df.sample(n=500, random_state=42).index.tolist()
    corpus_df = df.sample(n=7000, random_state=42)
    corpus_indices = corpus_df.index.tolist()
    corpus_indices = list(set(corpus_indices).difference(set(query_indices)))
    corpus_df = df.iloc[corpus_indices]
    corpus_embeddings = baseline_embeddings[corpus_indices]
    relevant_docs_map = {}
    top_k = 100
    similarity_threshold = 0.98
    print(f"Baseline:\nENCODER: {baseline_enc}_{baseline_enc_version}\nFIELD: {baseline_field}\n")
    print(f"Evaluating:\nENCODER: {encoder}_{encoder_version}\nFIELD: {field}\n")
    ignore_query_indices = set()
    for q_idx in query_indices:
        query_embedding = baseline_embeddings[q_idx]
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = cos_scores.topk(k=top_k, sorted=True)
        top_indices = top_results.indices
        top_values = top_results.values
        passed_threshold_mask = top_values >= similarity_threshold
        filtered_indices = top_indices[passed_threshold_mask].tolist()
        actual_indices = [corpus_indices[i] for i in filtered_indices]
        if len(actual_indices) == 0:
            ignore_query_indices.add(q_idx)
            continue
        relevant_docs_map[q_idx] = set(actual_indices)
    query_indices = [q for q in query_indices if q not in ignore_query_indices]
    corpus = {}
    for idx, row in corpus_df.iterrows():
        doc_id = f"d{idx}"
        corpus[doc_id] = row[field]

    queries = {}
    for q_idx in query_indices:
        query_id = f"q{q_idx}"
        queries[query_id] = df.loc[q_idx, field]

    relevant_docs = {}

    for q_idx in query_indices:
        query_id = f"q{q_idx}"
        doc_id_set = set()
        for doc_idx in relevant_docs_map[q_idx]:
            doc_id = f"d{doc_idx}"
            doc_id_set.add(doc_id)
        relevant_docs[query_id] = doc_id_set
    k_values = list(range(5, 110, 10))
    k_values.append(1)
    k_values.append(10)
    k_values.append(20)
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,               # dict: query_id -> query_text
        corpus=corpus,                 # dict: doc_id -> doc_text
        relevant_docs=relevant_docs,   # dict: query_id -> set(doc_ids)
        show_progress_bar=True,
        mrr_at_k=k_values,
        ndcg_at_k=k_values,
        accuracy_at_k=k_values,
        precision_recall_at_k=k_values,
        map_at_k=k_values,
        corpus_chunk_size=len(df),
    )
    model_field2 = load_encoder(encoder, encoder_version, model_to_load)
    eval_result = ir_evaluator(model_field2.model)
    metrics = [
        'mrr',
        'ndcg',
        'accuracy',
        'precision',
        'recall',
        'map'
    ]
    results = {}
    for metric in metrics:
        results[metric] = {}
    for k in k_values:
        k_str = str(k)
        for metric in metrics:
            results[metric][k_str] = eval_result[f"cosine_{metric}@{k}"]
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{encoder}_{encoder_version}_{field}_results.json")
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)
    print(f"Results saved successfully to {output_file}")
