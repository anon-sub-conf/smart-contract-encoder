from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable
import json
import os
import re
import math
import xml.etree.ElementTree as ET
from sentence_transformers import util
from smart_contract_encoder.encoder import *
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from smart_contract_encoder.load_data import *

def create_nicad_query_dataset(
    split,
    encoder,
    encoder_version,
    model_to_load=None,
    mapping_path="query_doc_map.json",  
):
    field = "func_code"
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

    # compute relevant docs per query (keeping ORDER from topk)
    ignore_query_indices = set()
    for q_idx in query_indices:
        query_embedding = baseline_embeddings[q_idx]
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = cos_scores.topk(k=top_k, sorted=True)
        top_indices = top_results.indices
        top_values = top_results.values

        # Keep only those above threshold, preserving the topk order
        filtered_pairs = []
        for j in range(len(top_indices)):
            score = float(top_values[j].item())
            if score >= similarity_threshold:
                corpus_pos = int(top_indices[j].item()) 
                doc_df_idx = corpus_indices[corpus_pos]
                filtered_pairs.append((doc_df_idx, score))

        if not filtered_pairs:
            ignore_query_indices.add(q_idx)
            continue

        relevant_docs_map[q_idx] = filtered_pairs 

    # Keep only queries that have at least one relevant doc
    print(len(query_indices))
    query_indices = [q for q in query_indices if q not in ignore_query_indices]
    print(len(query_indices))
    # output directories
    q_dir = Path("queries")
    d_dir = Path("documents")
    q_dir.mkdir(parents=True, exist_ok=True)
    d_dir.mkdir(parents=True, exist_ok=True)

    # write documents and build index->id map
    doc_idx_to_id = {}
    for i, idx in enumerate(corpus_df.index, start=1):
        doc_id = f"d_{i}"
        doc_idx_to_id[idx] = doc_id
        content = corpus_df.loc[idx, field]
        (d_dir / f"{doc_id}.sol").write_text(str(content) if content is not None else "", encoding="utf-8")

    # write queries and build index->id map
    query_idx_to_id = {}
    for i, q_idx in enumerate(query_indices, start=1):
        q_id = f"q_{i}"
        query_idx_to_id[q_idx] = q_id
        content = df.loc[q_idx, field]
        (q_dir / f"{q_id}.sol").write_text(str(content) if content is not None else "", encoding="utf-8")

    # build relevant_docs with order (IDs only, sorted by similarity desc)
    relevant_docs = {}
    for q_idx in query_indices:
        q_id = query_idx_to_id[q_idx]
        pairs = [(doc_idx_to_id[d_idx], sim) for (d_idx, sim) in relevant_docs_map[q_idx] if d_idx in doc_idx_to_id]
        pairs.sort(key=lambda x: -x[1]) 
        relevant_docs[q_id] = [doc_id for (doc_id, _) in pairs]

    Path(mapping_path).write_text(json.dumps(relevant_docs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Ground truth mapping JSON saved to {mapping_path}")



Q_RE = re.compile(r"(?:^|/)(q_\d+)\.sol$")
D_RE = re.compile(r"(?:^|/)(d_\d+)\.sol$")

def parse_similarity(val: str) -> float:
    if val is None:
        return 0.0
    s = val.strip()
    if s.endswith("%"):
        s = s[:-1]
    return float(s)

def extract_ids(path: str) -> Tuple[str, str]:
    q = Q_RE.search(path)
    if q:
        return q.group(1), None
    d = D_RE.search(path)
    if d:
        return None, d.group(1)
    return None, None

def build_mapping(xml_path: str) -> Dict[str, List[str]]:
    """
    Parse .xml output file produced by solidity-nicad 
    and collect the query to document mappings
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    best_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    for clone in root.findall(".//clone"):
        sim = parse_similarity(clone.get("similarity"))
        sources = clone.findall("./source")
        q_ids = []
        d_ids = []
        for s in sources:
            file_attr = s.get("file", "")
            q_id, d_id = extract_ids(file_attr)
            if q_id:
                q_ids.append(q_id)
            if d_id:
                d_ids.append(d_id)
        for q in q_ids:
            for d in d_ids:
                prev = best_scores[q].get(d, -1.0)
                if sim > prev:
                    best_scores[q][d] = sim

    result: Dict[str, List[str]] = {}
    for q, d_map in best_scores.items():
        ordered = sorted(d_map.items(), key=lambda kv: (-kv[1], kv[0]))
        result[q] = [doc_id for doc_id, _ in ordered]
    return result

def evaluate_nicad_results():
    pred = build_mapping("nicad_results.xml")
    with open("query_doc_map.json", "r", encoding="utf-8") as f:
        gt = json.load(f)
    k_vals = [1, 5, 10, 15, 20, 25, 35, 45, 55, 65, 75, 85, 95, 105]
    metrics = evaluate_at_cutoffs(gt, pred, k_vals)
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    with open("./results/nicad_untrained_func_code_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Wrote metrics to ./results/nicad_untrained_func_code_results.json")


def dcg_at_k(pred: List[str], rel_set: set, k: int) -> float:
    dcg = 0.0
    for i, doc in enumerate(pred[:k], start=1):
        rel = 1.0 if doc in rel_set else 0.0
        if rel:
            dcg += 1.0 / math.log2(i + 1.0)
    return dcg

def idcg_at_k(num_rel: int, k: int) -> float:
    ideal_hits = min(num_rel, k)
    return sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_hits + 1))

def ndcg_at_k(pred: List[str], rel_set: set, k: int) -> float:
    ideal = idcg_at_k(len(rel_set), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(pred, rel_set, k) / ideal

def precision_at_k(pred: List[str], rel_set: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for d in pred[:k] if d in rel_set)
    return hits / float(k)

def recall_at_k(pred: List[str], rel_set: set, k: int) -> float:
    denom = len(rel_set)
    if denom == 0:
        return 0.0
    hits = sum(1 for d in pred[:k] if d in rel_set)
    return hits / float(denom)

def ap_at_k(pred: List[str], rel_set: set, k: int) -> float:
    if len(rel_set) == 0:
        return 0.0
    ap_sum = 0.0
    hit_count = 0
    for i, d in enumerate(pred[:k], start=1):
        if d in rel_set:
            hit_count += 1
            ap_sum += hit_count / float(i)
    return ap_sum / float(len(rel_set))

def rr_at_k(pred: List[str], rel_set: set, k: int) -> float:
    for i, d in enumerate(pred[:k], start=1):
        if d in rel_set:
            return 1.0 / float(i)
    return 0.0

def accuracy_at_k(pred: List[str], rel_set: set, k: int) -> float:
    return 1.0 if any(d in rel_set for d in pred[:k]) else 0.0

def evaluate_at_cutoffs(
    gt: Dict[str, List[str]],
    pred: Dict[str, List[str]],
    cutoffs: Iterable[int],
) -> Dict[str, Dict[str, float]]:
    """
    gt: { q: [d1, d2, ...] } ground truth (ordered list; we use as a set for binary relevance)
    pred: { q: [d1, d2, ...] } predictions (ordered list; duplicates ignored naturally)
    """
    # Only consider queries with at least one relevant doc
    queries = [q for q, docs in gt.items() if len(docs) > 0]
    metrics = { 
        "mrr": {}, "ndcg": {}, "accuracy": {}, "precision": {}, "recall": {}, "map": {}
    }
    cutoffs = list(cutoffs)

    for k in cutoffs:
        mrr_sum = 0.0
        ndcg_sum = 0.0
        acc_sum = 0.0
        prec_sum = 0.0
        rec_sum = 0.0
        map_sum = 0.0

        for q in queries:
            rel_set = set(gt[q])
            pred_list = pred.get(q, [])

            seen = set()
            dedup_pred = []
            for d in pred_list:
                if d not in seen:
                    seen.add(d)
                    dedup_pred.append(d)

            mrr_sum += rr_at_k(dedup_pred, rel_set, k)
            ndcg_sum += ndcg_at_k(dedup_pred, rel_set, k)
            acc_sum += accuracy_at_k(dedup_pred, rel_set, k)
            prec_sum += precision_at_k(dedup_pred, rel_set, k)
            rec_sum += recall_at_k(dedup_pred, rel_set, k)
            map_sum += ap_at_k(dedup_pred, rel_set, k)

        denom = float(len(queries)) if queries else 1.0
        metrics["mrr"][str(k)] = mrr_sum / denom
        metrics["ndcg"][str(k)] = ndcg_sum / denom
        metrics["accuracy"][str(k)] = acc_sum / denom
        metrics["precision"][str(k)] = prec_sum / denom
        metrics["recall"][str(k)] = rec_sum / denom
        metrics["map"][str(k)] = map_sum / denom

    return metrics
