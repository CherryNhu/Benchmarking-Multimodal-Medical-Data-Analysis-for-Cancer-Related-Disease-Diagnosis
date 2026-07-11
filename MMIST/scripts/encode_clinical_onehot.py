"""
One-Hot Encode Clinical Variables → .pt Files
==============================================
Reads clinical.csv and one-hot encodes 11 specified variables for each patient.
Saves each patient as a {case_id}.pt file (768-dim float32 tensor, zero-padded).
Distributes files into feat/Clinical/train|val|test/ matching Clinical_1/ structure.
"""

import os
import shutil
from pathlib import Path

import pandas as pd
import torch

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST')
CSV_PATH = BASE_DIR / 'feat' / 'clinical.csv'

# Reference folder to determine which case_id goes to which split
REFERENCE_DIR = BASE_DIR / 'feat' / 'Clinical_1'

# Output directories
OUTPUT_DIR = BASE_DIR / 'feat' / 'Clinical'
TRAIN_DIR = OUTPUT_DIR / 'train'
VAL_DIR = OUTPUT_DIR / 'val'
TEST_DIR = OUTPUT_DIR / 'test'

# Target tensor dimension (must match model expectation)
TARGET_DIM = 768

# The 11 variables to one-hot encode
CATEGORICAL_COLS = [
    'cancer_history',
    'ajcc_path_tumor_pt',
    'ajcc_path_nodes_pn',
    'ajcc_clin_metastasis_cm',
    'ajcc_path_metastasis_pm',
    'ajcc_path_tumor_stage',
]

BINARY_COLS = [
    'race_Asian',
    'race_Black or African American',
    'race_Hispanic or Latino',
    'race_White',
    'race_other',
]

ALL_COLS = CATEGORICAL_COLS + BINARY_COLS


def get_split_mapping(reference_dir: Path) -> dict:
    """Read Clinical_1/ folder structure to build case_id -> split mapping."""
    mapping = {}
    for split in ['train', 'val', 'test']:
        split_dir = reference_dir / split
        if split_dir.exists():
            for f in split_dir.glob('*.pt'):
                case_id = f.stem  # filename without .pt
                mapping[case_id] = split
    return mapping


def encode_clinical_data(csv_path: Path) -> dict:
    """
    One-hot encode the 11 clinical variables for each patient.
    Returns: dict of {case_id: torch.Tensor (shape [TARGET_DIM])}
    """
    df = pd.read_csv(csv_path)
    
    # Process categorical columns: fill NaN with 'missing', treat -1 as 'missing'
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(-999)  # temporary marker
        df[col] = df[col].apply(lambda x: 'missing' if x == -999 or x == -1.0 or x == -1 else str(int(x) if isinstance(x, float) else x))
    
    # Process binary columns: fill NaN with 0, treat -1 as 0 (missing data)
    for col in BINARY_COLS:
        df[col] = df[col].fillna(0)
        df[col] = df[col].apply(lambda x: 0 if x == -1 else int(x))
    
    # One-hot encode categorical columns
    onehot_dfs = []
    for col in CATEGORICAL_COLS:
        dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
        # Sort columns for consistency
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        onehot_dfs.append(dummies)
    
    # Add binary columns as-is
    binary_df = df[BINARY_COLS].astype(float)
    onehot_dfs.append(binary_df)
    
    # Concatenate all encoded features
    encoded_df = pd.concat(onehot_dfs, axis=1)
    
    feature_dim = encoded_df.shape[1]
    print(f"One-hot encoded feature dimension: {feature_dim}")
    print(f"Columns: {list(encoded_df.columns)}")
    
    if feature_dim > TARGET_DIM:
        raise ValueError(
            f"Encoded features ({feature_dim}) exceed target dim ({TARGET_DIM}). "
            f"Cannot zero-pad."
        )
    
    # Build per-patient tensors
    patient_tensors = {}
    for idx, row in df.iterrows():
        case_id = row['case_id']
        features = torch.tensor(encoded_df.iloc[idx].values, dtype=torch.float32)
        
        # Zero-pad to TARGET_DIM
        if feature_dim < TARGET_DIM:
            padding = torch.zeros(TARGET_DIM - feature_dim, dtype=torch.float32)
            features = torch.cat([features, padding])
        
        patient_tensors[case_id] = features
    
    return patient_tensors


def main():
    # Setup output directories
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory: {d}")
    
    # Get split mapping from Clinical_1
    split_mapping = get_split_mapping(REFERENCE_DIR)
    print(f"\nSplit mapping loaded: {len(split_mapping)} case_ids")
    print(f"  train: {sum(1 for v in split_mapping.values() if v == 'train')}")
    print(f"  val:   {sum(1 for v in split_mapping.values() if v == 'val')}")
    print(f"  test:  {sum(1 for v in split_mapping.values() if v == 'test')}")
    
    # Encode clinical data
    print("\nEncoding clinical variables...")
    patient_tensors = encode_clinical_data(CSV_PATH)
    print(f"Encoded {len(patient_tensors)} patients")
    
    # Save to split directories
    split_dirs = {'train': TRAIN_DIR, 'val': VAL_DIR, 'test': TEST_DIR}
    saved_counts = {'train': 0, 'val': 0, 'test': 0}
    skipped = 0
    
    for case_id, tensor in patient_tensors.items():
        split = split_mapping.get(case_id)
        if split is None:
            print(f"  Warning: {case_id} not found in Clinical_1 splits, skipping")
            skipped += 1
            continue
        
        save_path = split_dirs[split] / f"{case_id}.pt"
        torch.save(tensor, save_path)
        saved_counts[split] += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"ENCODING COMPLETE")
    print(f"  Train: {saved_counts['train']} files")
    print(f"  Val:   {saved_counts['val']} files")
    print(f"  Test:  {saved_counts['test']} files")
    print(f"  Skipped: {skipped} (not in Clinical_1)")
    print(f"  Tensor shape: [{TARGET_DIM}]")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
