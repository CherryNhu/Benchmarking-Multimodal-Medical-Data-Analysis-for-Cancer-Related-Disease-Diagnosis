import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION (Setup your paths here)
# ==========================================
CSV_PATH = '/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/clinical.csv'
SOURCE_DIR = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/all/Clinical')

# Target directories
BASE_TARGET_DIR = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/Clinical')
TRAIN_DIR = BASE_TARGET_DIR / 'train'
VAL_DIR = BASE_TARGET_DIR / 'val'
TEST_DIR = BASE_TARGET_DIR / 'test'

def setup_directories():
    """Creates target directories if they don't exist, clearing them if necessary."""
    for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def copy_files(case_ids, target_dir):
    """Copies .pt files corresponding to case_ids to the target directory."""
    missing_files = 0
    copied_files = 0

    for case_id in case_ids:
        # Assuming the filename format is exactly '{case_id}.pt'
        file_name = f"{case_id}.pt"
        src_file = SOURCE_DIR / file_name
        dest_file = target_dir / file_name

        if src_file.exists():
            shutil.copy2(src_file, dest_file) # copy2 preserves metadata
            copied_files += 1
        else:
            print(f"Warning: File not found -> {src_file}")
            missing_files += 1
            
    print(f"Successfully copied {copied_files} files to {target_dir.name}.")
    if missing_files > 0:
        print(f"Total missing files: {missing_files}")

def main():
    setup_directories()

    print("\nReading CSV and resolving splits...")
    # Load the CSV
    df = pd.read_csv(CSV_PATH)

    # 1. Extract the predefined 'test' split
    test_cases = df[df['Split'] == 'test']['case_id'].tolist()

    # 2. Extract the predefined 'train' split
    original_train_cases = df[df['Split'] == 'train']['case_id'].tolist()

    # 3. Dynamically split 20% of 'train' into 'val'
    # random_state ensures reproducibility (the split is exactly the same every time you run it)
    train_cases, val_cases = train_test_split(
        original_train_cases, 
        test_size=0.20, 
        random_state=42 
    )

    print(f"Distribution Strategy:")
    print(f"- Train: {len(train_cases)} files")
    print(f"- Val: {len(val_cases)} files")
    print(f"- Test: {len(test_cases)} files\n")

    print("Executing file routing...")
    copy_files(train_cases, TRAIN_DIR)
    copy_files(val_cases, VAL_DIR)
    copy_files(test_cases, TEST_DIR)
    
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()