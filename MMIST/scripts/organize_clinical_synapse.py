import pandas as pd
from pathlib import Path
import shutil

def organize_train_test_splits_by_moving(base_dir_path):
    # 1. Define Paths
    base_dir = Path(base_dir_path)
    clinical_dir = base_dir / "Clinical"
    csv_path = base_dir / "cleaned_split_dataset.csv"
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    
    # Create target directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load the CSV
    print(f"Loading split data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return

    # 3. Handle Duplicate Patient Rows
    unique_patients = df.drop_duplicates(subset=['dmp_pt_id', 'split'])
    
    missing_files = []
    success_count = 0
    already_moved_count = 0
    
    # 4. Process the files
    print("Moving files...")
    for _, row in unique_patients.iterrows():
        patient_id = row['dmp_pt_id']
        split_label = row['split']
        
        source_file = clinical_dir / f"{patient_id}.pt"
        dest_dir = train_dir if split_label == 'train' else test_dir
        dest_file = dest_dir / f"{patient_id}.pt"
        
        # Check if the file is still in the Clinical folder
        if source_file.exists():
            # --- NEW: Use shutil.move instead of copy2 ---
            shutil.move(str(source_file), str(dest_file))
            success_count += 1
        else:
            # If it's not in Clinical, check if it was ALREADY moved to the destination
            if dest_file.exists():
                already_moved_count += 1
            else:
                missing_files.append(patient_id)
            
    # 5. Reporting
    print(f"\n--- Operation Complete ---")
    print(f"Successfully moved: {success_count} files.")
    if already_moved_count > 0:
        print(f"Skipped {already_moved_count} files (already in destination folders).")
    
    if missing_files:
        print(f"Warning: {len(missing_files)} files were missing completely.")

if __name__ == "__main__":
    TARGET_DIRECTORY = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE"
    organize_train_test_splits_by_moving(TARGET_DIRECTORY)