from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from sentence_transformers import util
from smart_contract_encoder.encoder import *
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from smart_contract_encoder.load_data import *


def create_nicad_query_dataset(
    split,
    encoder,
    encoder_version,
    model_to_load=None,
    field="func_code",
    queries_dir="queries",
    documents_dir="documents",
):
    df = load_dataset(file_type="merged", split=split)
    baseline_enc = "sentence_encoder"
    baseline_enc_version = "untrained"
    baseline_field = "func_documentation"
    baseline_embeddings = load_dataset(
        file_type="embeddings",
        split=split,
        encoder=baseline_enc,
        encoder_version=baseline_enc_version,
        field=baseline_field,
    )
    baseline_embeddings = np.stack(baseline_embeddings["embeddings"].to_list())

    query_indices = df.sample(n=500, random_state=42).index.tolist()
    corpus_df = df.sample(n=7000, random_state=42)
    corpus_indices = corpus_df.index.tolist()
    corpus_indices = list(set(corpus_indices).difference(set(query_indices)))
    corpus_df = df.iloc[corpus_indices]
    corpus_embeddings = baseline_embeddings[corpus_indices]

    # Retrieval params
    relevant_docs_map = {}
    top_k = 100
    similarity_threshold = 0.98

    print(
        f"Baseline:\nENCODER: {baseline_enc}_{baseline_enc_version}\nFIELD: {baseline_field}\n"
    )
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

    # Keep only queries that have at least one relevant document
    query_indices = [q for q in query_indices if q not in ignore_query_indices]

    # Create output directories 
    q_dir = Path(queries_dir)
    d_dir = Path(documents_dir)
    q_dir.mkdir(parents=True, exist_ok=True)
    d_dir.mkdir(parents=True, exist_ok=True)

    # Write documents (corpus) to files and build ID remap
    # Map from original corpus df index -> new document id like "d_1"
    doc_idx_to_new_id = {}
    for i, idx in enumerate(corpus_df.index, start=1):
        new_doc_id = f"d_{i}"
        doc_idx_to_new_id[idx] = new_doc_id

        content = corpus_df.loc[idx, field]
        with open(d_dir / f"{new_doc_id}.sol", "w", encoding="utf-8") as f:
            f.write(str(content) if content is not None else "")

    # Write queries to files and build ID remap 
    # Map from original query df index -> new query id like "q_1"
    query_idx_to_new_id = {}
    for i, q_idx in enumerate(query_indices, start=1):
        new_query_id = f"q_{i}"
        query_idx_to_new_id[q_idx] = new_query_id

        content = df.loc[q_idx, field]
        with open(q_dir / f"{new_query_id}.sol", "w", encoding="utf-8") as f:
            f.write(str(content) if content is not None else "")

