from smart_contract_encoder.data_curation import *
from smart_contract_encoder.encoder import *
from smart_contract_encoder.training_data_creation import *
from smart_contract_encoder.models.sentence_encoder import *
from smart_contract_encoder.eval_data_creation import *
from smart_contract_encoder.nicad_eval_data_creation import *
from smart_contract_encoder.graphs import *
import numpy as np
from sentence_transformers import util
import json

def main():
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


if __name__ == '__main__':
    main()