# from datasets import load_dataset
import datasets
import json
import pandas as pd
from tqdm import tqdm
import datasets
from pathlib import Path
from smart_contract_encoder.utils import *

SOURCE_DATASET_PATH = (Path(__file__).resolve().parent.parent.parent / "source_dataset")
DATA_DIR = (Path(__file__).resolve().parent.parent.parent / "data")

FIELD_VALS = ["func_code", "code", "tac_code", "opcode", "func_documentation", "signature", "func_name", "func_documentation_func_code"]
SPLIT_VALS = ["train", "test", "validation"]

def load_dataset_local(path: str):
    dataset = datasets.load_dataset("json", path, split="train")
    return dataset

def get_path(file_type: str, split: str, encoder: str = None,
             encoder_version: str = None, field: str = None,
             training_dataset_type: str = None) -> Path:
    if split not in SPLIT_VALS:
        raise Exception(f'Split must be one of {", ".join(SPLIT_VALS)}')
    if encoder is not None and encoder not in ENC_VALS:
        raise Exception(f'Encoder must be one of {", ".join(ENC_VALS)}')
    if encoder_version is not None and encoder_version not in ENC_VERSION:
        raise Exception(f'Encoder version must be one of {", ".join(ENC_VERSION)}')
    if field is not None and field not in FIELD_VALS:
        raise Exception(f'Field must be one of {", ".join(FIELD_VALS)}')
    if training_dataset_type is not None and training_dataset_type not in INPUT_TYPE:
        raise Exception(f'Training dataset must be one of {", ".join(INPUT_TYPE)}')
    match file_type:
        case "addresses":
            return Path.joinpath(DATA_DIR, f"{split}_contract_addresses.csv")
        case "source":
            return Path.joinpath(SOURCE_DATASET_PATH, split)
        case "decompiled":
            return Path.joinpath(DATA_DIR,"decomp_datasets", f"{split}_contract_addresses_decompiled.pkl")
        case "merged":
            return Path.joinpath(DATA_DIR, f"{split}_merged.pkl")
        case "bytecode":
            return Path.joinpath(DATA_DIR, "bytecode_dataset", f"{split}_contract_addresses_bytecode.csv")
        case "embeddings":
            return Path.joinpath(DATA_DIR, f"{split}_{encoder}_{encoder_version}_{field}_embeddings.pkl")
        case "training":
            return Path.joinpath(DATA_DIR, f"{split}_{field}_{training_dataset_type}.json")
        case _:
            raise Exception(f"Unknown file type {file_type}")


def save_dataset(dataset, file_type: str, split: str,  encoder: str = None,
                 encoder_version: str = None, field: str = None,
                 training_dataset_type: str = None) -> None:
    filename = get_path(file_type, split, encoder, encoder_version,
                        field, training_dataset_type)
    suffix = str(filename).split('.')[-1]
    if suffix == "csv":
        dataset.to_csv(filename, index=False)
    elif suffix == "pkl":
        dataset.to_pickle(filename)
    elif suffix == "json":
        with open(filename, "w") as f:
            json.dump(dataset, f, indent=2)
    else:
        raise Exception(f"Unknown file {filename}")
    print(f"Dataset saved to {filename}")


def load_dataset(file_type: str, split: str,  encoder: str = None,
                 encoder_version: str = None, field: str = None,
                 training_dataset_type: str = None):
    filename = get_path(file_type, split, encoder, encoder_version, field, training_dataset_type)
    if not filename.exists() and not filename.is_dir():
        return None
    if file_type == "source":
        dataset = pd.DataFrame()
        for file in tqdm(filename.iterdir()):
            data_part = pd.read_parquet(file)
            dataset = pd.concat([dataset, data_part])
        dataset.reset_index(drop=True, inplace=True)
        return dataset
    suffix = str(filename).split('.')[-1]
    if suffix == "csv":
        return pd.read_csv(filename)
    elif suffix == "pkl":
        return pd.read_pickle(filename)
    elif suffix == "json":
        train_data = datasets.load_dataset('json', data_files=str(filename), field="data", split="train")
        eval_data = datasets.load_dataset('json', data_files=str(filename), field="eval", split="train")
        return train_data, eval_data
    else:
        raise Exception(f"Unknown file {filename}")

