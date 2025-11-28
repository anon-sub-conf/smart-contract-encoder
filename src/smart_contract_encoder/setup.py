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
    
    # process / create datasets
    process_datasets("train", None)
    process_datasets("test", None)
    test_dataset = load_dataset("merged", split="test")
    test_dataset[['func_code']].to_csv("functions.csv", index=False)
    create_translation_pairs_dataset("code")
    create_translation_pairs_dataset("tac_code")

    baseline_enc = "sentence_encoder"
    baseline_enc_version = "untrained"
    baseline_field = "func_documentation"
    baseline_embeddings = create_embeddings("train", encoder=baseline_enc, encoder_version=baseline_enc_version, field=baseline_field)

    create_pairs_dataset("train", "code")
    create_pairs_dataset("train", "tac_code")
    create_nicad_query_dataset("test", "nicad", "untrained")



if __name__ == '__main__':
    main()
