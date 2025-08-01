import pandas as pd
from tqdm import tqdm
import regex as re
from typing import List
from smart_contract_encoder.load_data import *
tqdm.pandas()

def filter_fields(df: pd.DataFrame, filters: List[str]) -> pd.DataFrame:
    for filter in filters:
        len_before = len(df)
        match filter:
            case "doc_type":
                df = filter_doc_type(df)
            case "language":
                df = filter_language(df)
            case "non_public":
                df = filter_non_public(df)
            case "duplicate_code_doc":
                df = filter_duplicate_code_docs(df)
            case "doc_has_no_letters":
                df = filter_doc(df)
            case "non_implementation":
                df = filter_non_impl_code(df)
            case "noop_fallback":
                df = filter_noop_fallback(df)
            case "short_doc":
                df = filter_short_docs(df)
            case "duplicate_doc_in_contract":
                df = filter_duplicate_docs_in_contract(df)
            case "extract_opcodes":
                df = extract_opcodes(df)
            case _:
                raise Exception(f"Unknown filter {filter}")
        print(f"{len_before - len(df)} rows removed by {filter}.")
    return df

def filter_doc_type(df: pd.DataFrame) -> pd.DataFrame:
    # Keeping only sources with NatSpecLine documentation type
    df = df[df['func_documentation_type'].isin({'NatSpecMultiLine', 'NatSpecSingleLine'})]
    return df

def filter_language(df: pd.DataFrame) -> pd.DataFrame:
    # Keeping only contracts written in Solidity
    df = df[df['language']=='Solidity']
    return df

def filter_non_public(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where the 'func_code' does not contain
    'public' or 'external'.
    """
    mask = df['func_code'].str.contains(r'\s(public|external)\s', case=True, na=False)
    df = df[mask]
    return df

def filter_duplicate_code_docs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates based on the combination of (func_documentation, func_code).
    Keep only the first occurrence.
    """
    df.drop_duplicates(subset=['func_documentation', 'func_code'], keep='first', inplace=True)
    return df


def filter_doc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where 'func_documentation' contains no letters
    """
    df = df[df['func_documentation'].str.contains(r'[A-Za-z]', na=False)]
    return df

def filter_non_impl_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where 'func_code' is not a function implementation
    """
    df = df[df['func_code'].str.contains(r'{', na=False)]
    return df

def filter_noop_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes likely no-op fallback functions:
    Those where func_code < 60 chars AND func_name is empty or "fallback".
    """
    df = df[~((df['func_code'].str.len()<60) & ((df['func_name'].str.len()==0) | (df['func_name'] == 'fallback')))]
    return df

def filter_short_docs(df: pd.DataFrame, min_length=70) -> pd.DataFrame:
    """
    Removes rows where 'func_documentation' is shorter than min_length.
    """
    df = df[df['func_documentation'].str.len() >= min_length]
    return df

def filter_duplicate_docs_in_contract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes repeated 'func_documentation' within the same contract_address.
    Drop all duplicates entirely if they appear more than once in the same contract.
    """
    df.drop_duplicates(subset=['func_documentation', 'contract_address'], keep=False, inplace=True)
    return df



# transform functions

def rename_fallback_functions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces 'func_name' value with 'fallback' when 'func_name' is empty string.
    """
    df['func_name'] = df['func_name'].progress_apply(lambda s: s if len(s) > 0 else "fallback")
    return df


def extract_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new column 'signature' by concatenating 'func_name'
    and parameter-list extracted from 'func_code'.
    """
    print("Extracting function signatures.")
    df['signature'] = df['func_name'] + df['func_code'].progress_apply(extract_parameter_types)
    return df

def _get_right_paren(s):
    count = 1
    for i,c in enumerate(s):
        if c == '(':
            count += 1
        elif c == ')':
            count -= 1
        if count == 0:
            return i

def extract_opcodes(df: pd.DataFrame) -> pd.DataFrame:
    df['opcode'] = df['tac_code'].progress_apply(extract_opcodes_from_row)
    return df

def extract_opcodes_from_row(tac_code: str) -> str:
    return ' '.join([k.split("=")[-1].strip().split(' ')[0] for k in tac_code.split('\n')])

def extract_parameter_types(func_code: str) -> str:
    """
    Extracts parameter types from the function definition in Solidity.
    Returns '(type1, type2, ...)' or '()' if none found.
    """
    lp = func_code.index('(')
    x = func_code[lp+1:]
    rp = _get_right_paren(x)
    x = x[:rp]
    params = x.split(',')
    types = []
    for param in params:
        cleaned_param = re.sub(r'/\*.*?\*/', '', param, flags=re.DOTALL).strip()
        if cleaned_param.startswith("mapping"):
            rp = cleaned_param.index(')')
            cleaned_param = cleaned_param[:rp+1]
        else:
            if len(cleaned_param) == 0:
                continue
            cleaned_param = cleaned_param.split()[0]
        types.append(cleaned_param)
    pars = f"({', '.join(types)})" if types else "()"
    return pars

def remove_uneeded_cols(df):
    cols_to_drop = ['class_code', 'license_type', 'file_path', 'language',
                    'contract_name', 'compiler_version', 'swarm_source',
                    'class_documentation', 'class_documentation_type',
                    'class_name', 'func_documentation_type']
    print(f"Dropping columns {cols_to_drop}")
    return df.drop(columns=cols_to_drop)


def remove_duplicates_by_normalised_md5(df, split):
    bytecode_df = load_dataset(file_type="bytecode", split=split)[['address', 'normalised_md5']]
    bytecode_df['address'] = '0x'+bytecode_df['address'].astype(str)
    len_before = df['contract_address'].nunique()
    merged = df.merge(bytecode_df[['address', 'normalised_md5']],
                      how='left',
                      left_on='contract_address',
                      right_on='address')
    merged.drop(columns='address', inplace=True, errors='ignore')
    merged.dropna(subset=['normalised_md5'], inplace=True)
    len_after = merged['contract_address'].nunique()
    merged.reset_index(drop=True, inplace=True)
    print(f"{len_before - len_after} rows removed by normalised_md5.")
    return merged

def curate_datasets(split, args=None):
    print(f"Curating the dataset for the {split} split")
    df = load_dataset(file_type="source", split=split)
    df = remove_duplicates_by_normalised_md5(df, split)
    df['func_documentation'] = df['func_documentation'].str.strip()
    df = filter_fields(df, [
        "language",
        "doc_type",
        "non_public",
        "doc_has_no_letters",
        "non_implementation",
        "noop_fallback",
        "short_doc",
        "duplicate_code_doc",
        "duplicate_doc_in_contract"
        ])
    df = rename_fallback_functions(df)
    df = extract_signatures(df)
    df = remove_uneeded_cols(df)
    df.reset_index(drop=True, inplace=True)
    print(df)
    return df

def process_datasets(split, args):
    src_df = curate_datasets(split, args)
    decomp_df = load_dataset(file_type="decompiled", split=split)
    decomp_df['func_name'] = decomp_df['func_name'].str[13:]
    decomp_df = decomp_df[decomp_df['code'].str.len() > 100]
    decomp_df['signature'] = decomp_df['func_name'] + decomp_df['code'].progress_apply(extract_parameter_types)
    src_df = src_df.merge(decomp_df[['contract_normalised_md5', 'signature', 'code', 'tac_code']],
                          how='left',
                          left_on=['normalised_md5', 'signature'],
                          right_on=['contract_normalised_md5', 'signature'])
    src_df.dropna(subset=['code'], inplace=True)
    src_df.drop(columns="contract_normalised_md5", inplace=True)
    src_df.reset_index(drop=True, inplace=True)
    src_df = filter_fields(src_df, [
        "extract_opcodes"
        ])
    save_dataset(src_df, file_type="merged", split=split)
    return src_df
