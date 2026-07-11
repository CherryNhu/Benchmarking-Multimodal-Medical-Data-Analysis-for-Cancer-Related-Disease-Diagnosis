#!/usr/bin/env python3
"""
Create stratified group k-fold splits for SYNAPSE dataset
Groups by patient ID (dmp_pt_id) and stratifies by label
Uses symlinks to avoid duplicating large files
Implements stratified group k-fold without sklearn to avoid dependency issues
"""

import os
import csv
import random
from pathlib import Path
from collections import defaultdict

# Paths
DATA_DIR = Path("/mmlab_students/storageStudents/nguyenvd/Nhulnq/RunBaseline/MMIST/feat/feat-SYNAPSE")
CSV_FILE = DATA_DIR / "cleaned_split_dataset.fixed.csv"
OUTPUT_DIR = DATA_DIR.parent / "feat-SYNAPSE_cv"

# Load data manually
data = []
with open(CSV_FILE, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

print(f"Total samples: {len(data)}")

# Get unique patients and their label (assuming all records of same patient have same label)
unique_patients = {}
for row in data:
    pt_id = row['dmp_pt_id']
    label = int(row['label'])
    if pt_id not in unique_patients:
        unique_patients[pt_id] = label

print(f"Unique patients: {len(unique_patients)}")

# Group patients by label
patients_by_label = defaultdict(list)
for pt_id, label in unique_patients.items():
    patients_by_label[label].append(pt_id)

print(f"Label 0: {len(patients_by_label[0])} patients")
print(f"Label 1: {len(patients_by_label[1])} patients")

# Simple stratified k-fold: shuffle within each label group and assign folds
random.seed(42)
k_folds = 5
patient_to_fold = {}

for label, patients in patients_by_label.items():
    # Shuffle patients with this label
    shuffled = patients.copy()
    random.shuffle(shuffled)
    
    # Divide into k folds
    fold_size = len(shuffled) // k_folds
    remainder = len(shuffled) % k_folds
    
    fold_idx = 0
    start_idx = 0
    for fold_num in range(k_folds):
        # Distribute remainder patients across first folds
        current_fold_size = fold_size + (1 if fold_num < remainder else 0)
        end_idx = start_idx + current_fold_size
        
        for pt_id in shuffled[start_idx:end_idx]:
            patient_to_fold[pt_id] = fold_num
        
        start_idx = end_idx

print(f"\nFold distribution:")
fold_counts = defaultdict(lambda: {'total': 0, 'label_0': 0, 'label_1': 0})
for pt_id, fold_idx in patient_to_fold.items():
    fold_counts[fold_idx]['total'] += 1
    if unique_patients[pt_id] == 0:
        fold_counts[fold_idx]['label_0'] += 1
    else:
        fold_counts[fold_idx]['label_1'] += 1

for fold_idx in range(k_folds):
    counts = fold_counts[fold_idx]
    print(f"  Fold {fold_idx}: {counts['total']} patients (label 0: {counts['label_0']}, label 1: {counts['label_1']})")

# Create directory structure
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Get all Clinical, CT, WSI files
clinical_train = {f.stem for f in (DATA_DIR / 'Clinical' / 'train').glob('*.pt')}
clinical_test = {f.stem for f in (DATA_DIR / 'Clinical' / 'test').glob('*.pt')}

# For each fold, create train/val/test split
for fold_idx in range(k_folds):
    print(f"\n--- Processing Fold {fold_idx} ---")
    
    fold_dir = OUTPUT_DIR / f"fold_{fold_idx}"
    fold_dir.mkdir(exist_ok=True)
    
    # Get test patients for this fold
    test_patients = {pt for pt, f in patient_to_fold.items() if f == fold_idx}
    train_patients = {pt for pt, f in patient_to_fold.items() if f != fold_idx}
    
    print(f"Test patients: {len(test_patients)}")
    print(f"Train patients: {len(train_patients)}")
    
    # Split train into train/val (80/20)
    train_list = sorted(list(train_patients))
    n_train = len(train_list)
    n_val = max(1, n_train // 5)
    
    # Use deterministic split based on fold_idx
    rng = random.Random(42 + fold_idx)
    indices = list(range(n_train))
    rng.shuffle(indices)
    
    val_patients = set(train_list[i] for i in indices[-n_val:])
    train_final_patients = set(train_list[i] for i in indices[:-n_val])
    
    print(f"  Train: {len(train_final_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
    
    # Create master_dataset.csv
    master_data = []
    
    # For Clinical modality
    for pt_id in train_final_patients:
        if pt_id in clinical_train or pt_id in clinical_test:
            master_data.append({
                'case_id': pt_id,
                'file_name': f"{pt_id}.pt",
                'Modality': 'Clinical',
                'Split': 'train',
                'label': unique_patients[pt_id]
            })
    
    for pt_id in val_patients:
        if pt_id in clinical_train or pt_id in clinical_test:
            master_data.append({
                'case_id': pt_id,
                'file_name': f"{pt_id}.pt",
                'Modality': 'Clinical',
                'Split': 'val',
                'label': unique_patients[pt_id]
            })
    
    for pt_id in test_patients:
        if pt_id in clinical_train or pt_id in clinical_test:
            master_data.append({
                'case_id': pt_id,
                'file_name': f"{pt_id}.pt",
                'Modality': 'Clinical',
                'Split': 'test',
                'label': unique_patients[pt_id]
            })
    
    master_csv = fold_dir / "master_dataset.csv"
    with open(master_csv, 'w', newline='') as f:
        if master_data:
            fieldnames = ['case_id', 'file_name', 'Modality', 'Split', 'label']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(master_data)
    
    print(f"Created {master_csv} with {len(master_data)} entries")
    
    # Create modality directories with symlinks
    modalities = ['Clinical', 'CT', 'WSI']
    
    for modality in modalities:
        modality_source = DATA_DIR / modality
        if not modality_source.exists():
            print(f"  Modality {modality} not found in source, skipping...")
            continue
        
        modality_dir = fold_dir / modality
        modality_dir.mkdir(exist_ok=True)
        
        # Create train/val/test subdirectories
        for split_type in ['train', 'val', 'test']:
            split_dir = modality_dir / split_type
            split_dir.mkdir(exist_ok=True)
        
        # Get patients for each split
        split_to_patients = {
            'train': train_final_patients,
            'val': val_patients,
            'test': test_patients
        }
        
        print(f"  Organizing {modality}...")
        
        if modality == 'Clinical':
            # Clinical files are in Clinical/train and Clinical/test
            for split_name, patients in split_to_patients.items():
                split_dir = modality_dir / split_name
                for pt_id in patients:
                    # Check in both train and test
                    source_file = None
                    if pt_id in clinical_train:
                        source_file = DATA_DIR / 'Clinical' / 'train' / f"{pt_id}.pt"
                    elif pt_id in clinical_test:
                        source_file = DATA_DIR / 'Clinical' / 'test' / f"{pt_id}.pt"
                    
                    if source_file and source_file.exists():
                        dest_file = split_dir / f"{pt_id}.pt"
                        if not dest_file.exists():
                            os.symlink(source_file, dest_file)
        
        elif modality == 'CT':
            # CT has directories per patient with multiple files inside
            ct_train_dir = DATA_DIR / 'CT' / 'train'
            ct_test_dir = DATA_DIR / 'CT' / 'test'
            
            # Find CT directories that match patients
            ct_dirs_train = {d.name: d for d in ct_train_dir.iterdir() if d.is_dir()}
            ct_dirs_test = {d.name: d for d in ct_test_dir.iterdir() if d.is_dir()}
            
            for split_name, patients in split_to_patients.items():
                split_dir = modality_dir / split_name
                for ct_dir_name, ct_source_dir in {**ct_dirs_train, **ct_dirs_test}.items():
                    # Create symlink to CT directory
                    dest_dir = split_dir / ct_dir_name
                    if not dest_dir.exists():
                        os.symlink(ct_source_dir, dest_dir)
        
        elif modality == 'WSI':
            # WSI files are pdl1_image_id.pt
            wsi_train_dir = DATA_DIR / 'WSI' / 'train'
            wsi_test_dir = DATA_DIR / 'WSI' / 'test'
            
            # Build mapping of pdl1_image_id to files
            wsi_train_files = {f.stem: f for f in wsi_train_dir.glob('*.pt')}
            wsi_test_files = {f.stem: f for f in wsi_test_dir.glob('*.pt')}
            
            # Get pdl1_image_ids for each patient and split
            for row in data:
                if row['dmp_pt_id'] in split_to_patients['train'] and row['pdl1_image_id']:
                    pdl1_id = row['pdl1_image_id']
                    if pdl1_id in wsi_train_files or pdl1_id in wsi_test_files:
                        source_file = wsi_train_files.get(pdl1_id) or wsi_test_files.get(pdl1_id)
                        dest_file = modality_dir / 'train' / f"{pdl1_id}.pt"
                        if not dest_file.exists():
                            os.symlink(source_file, dest_file)
                
                if row['dmp_pt_id'] in split_to_patients['val'] and row['pdl1_image_id']:
                    pdl1_id = row['pdl1_image_id']
                    if pdl1_id in wsi_train_files or pdl1_id in wsi_test_files:
                        source_file = wsi_train_files.get(pdl1_id) or wsi_test_files.get(pdl1_id)
                        dest_file = modality_dir / 'val' / f"{pdl1_id}.pt"
                        if not dest_file.exists():
                            os.symlink(source_file, dest_file)
                
                if row['dmp_pt_id'] in split_to_patients['test'] and row['pdl1_image_id']:
                    pdl1_id = row['pdl1_image_id']
                    if pdl1_id in wsi_train_files or pdl1_id in wsi_test_files:
                        source_file = wsi_train_files.get(pdl1_id) or wsi_test_files.get(pdl1_id)
                        dest_file = modality_dir / 'test' / f"{pdl1_id}.pt"
                        if not dest_file.exists():
                            os.symlink(source_file, dest_file)

print(f"\n✓ Created fold structure in {OUTPUT_DIR}")
print("✓ Folds are ready for training with stratified group k-fold (k=5)")
