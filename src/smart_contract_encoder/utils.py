from pathlib import Path

ENC_VALS = ["sentence_encoder", "codet5", "ngram"]
ENC_VERSION = ["finetuned", "untrained"]

MODEL_DIR = (Path(__file__).resolve().parent.parent.parent / "all-mpnet-base-v2")
INPUT_LEVEL = ["code", "tac_code", "func_documentation"]
INPUT_TYPE = ["translation_pairs", "positive_pairs"]
