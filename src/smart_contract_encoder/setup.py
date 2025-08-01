from smart_contract_encoder.data_curation import *
from smart_contract_encoder.encoder import *
from smart_contract_encoder.training_data_creation import *
from smart_contract_encoder.models.sentence_encoder import *
from smart_contract_encoder.eval_data_creation import *
from smart_contract_encoder.graphs import *
import numpy as np
from sentence_transformers import util
import json

def main():
    # process / create datasets
    process_datasets("train", None)
    process_datasets("test", None)
    create_translation_pairs_dataset("code")
    create_translation_pairs_dataset("tac_code")

    baseline_enc = "sentence_encoder"
    baseline_enc_version = "untrained"
    baseline_field = "func_documentation"
    baseline_embeddings = create_embeddings("train", encoder=baseline_enc, encoder_version=baseline_enc_version, field=baseline_field)

    create_pairs_dataset("train", "code")
    create_pairs_dataset("train", "tac_code")

    # finetune different models
    field = "code"
    training_dataset_type="positive_pairs"
    train, eval = load_dataset(file_type="training", field=field, split="train", training_dataset_type=training_dataset_type)
    encoder = SentenceEncoder(load=False, input_level=field, input_type=training_dataset_type)
    encoder.finetune_pairs(eval, train)

    field = "tac_code"
    training_dataset_type="positive_pairs"
    train, eval = load_dataset(file_type="training", field=field, split="train", training_dataset_type=training_dataset_type)
    encoder = SentenceEncoder(load=False, input_level=field, input_type=training_dataset_type)
    encoder.finetune_pairs(eval, train)

    field = "code"
    training_dataset_type="translation_pairs"
    train, eval = load_dataset(file_type="training", field=field, split="train", training_dataset_type=training_dataset_type)
    encoder = SentenceEncoder(load=False, input_level=field, input_type=training_dataset_type)
    encoder.finetune_pairs(eval, train)

    field = "tac_code"
    training_dataset_type="translation_pairs"
    train, eval = load_dataset(file_type="training", field=field, split="train", training_dataset_type=training_dataset_type)
    encoder = SentenceEncoder(load=False, input_level=field, input_type=training_dataset_type)
    encoder.finetune_pairs(eval, train)

    field = "code"
    training_dataset_type="positive_pairs"
    train, eval = load_dataset(file_type="training", field=field, split="train", training_dataset_type=training_dataset_type)
    encoder = SentenceEncoder(load=True, model_to_load="all-mpnet-base-v2/code_translation_pairs_32_3000_50_50", input_level=field, input_type=training_dataset_type)
    encoder.finetune_pairs(eval, train)

    field = "tac_code"
    training_dataset_type="positive_pairs"
    train, eval = load_dataset(file_type="training", field=field, split="train", training_dataset_type=training_dataset_type)
    encoder = SentenceEncoder(load=True, model_to_load="all-mpnet-base-v2/tac_code_translation_pairs_32_3000_50_50", input_level=field, input_type=training_dataset_type)
    encoder.finetune_pairs(eval, train)


    # evaluate all different models
    baseline_enc = "sentence_encoder"
    baseline_enc_version = "untrained"
    baseline_field = "func_documentation"
    baseline_embeddings = create_embeddings("test", encoder=baseline_enc, encoder_version=baseline_enc_version, field=baseline_field)
    create_query_dataset("test", "sentence_encoder", "untrained", "code")
    create_query_dataset("test", "sentence_encoder", "untrained", "tac_code")
    create_query_dataset("test", "sentence_encoder", "untrained", "func_code")
    create_query_dataset("test", "sentence_encoder", "finetuned", "code", "all-mpnet-base-v2/code_translation_pairs_32_3000_50_50")
    create_query_dataset("test", "sentence_encoder", "finetuned", "tac_code", "all-mpnet-base-v2/tac_code_translation_pairs_32_3000_50_50")
    create_query_dataset("test", "ngram", "untrained", "opcode")

    # create evaluation graphs
    eval_graphs()
    dist_graphs()


if __name__ == '__main__':
    main()
