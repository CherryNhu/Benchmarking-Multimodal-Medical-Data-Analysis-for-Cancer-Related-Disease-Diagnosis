import pandas as pd
import shutil
from pathlib import Path

def move_wsi_dataset(raw_dir: str, dest_dir: str, csv_path: str):
    """
    Moves WSI .pt files into train/test folders based on a CSV.
    WARNING: This modifies the original directory structure permanently.
    """
    raw_path = Path(raw_dir)
    dest_path = Path(dest_dir)
    
    # 1. Read CSV with strict string types for the identifier
    print(f"Loading split mapping from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={'slide_id': str})
    
    # Clean the data
    df = df.dropna(subset=['slide_id'])
    df['slide_id'] = df['slide_id'].str.strip()
    
    # Create the mapping dictionary
    split_mapping = dict(zip(df['slide_id'], df['split']))

    # 2. Process the WSI directory
    print(f"Moving .pt files from {raw_dir}...")
    success_count = 0
    missing_count = 0

    for file_path in raw_path.glob('*.pt'):
        slide_id = file_path.stem
        
        if slide_id not in split_mapping:
            print(f"  [!] Warning: slide_id {slide_id} not found in CSV. Leaving file in place.")
            missing_count += 1
            continue
            
        split_type = split_mapping[slide_id]
        
        # Define target directory
        target_dir = dest_path / split_type
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_file_path = target_dir / file_path.name
        
        # 3. Move the file
        if target_file_path.exists():
            print(f"  [!] Skip: {file_path.name} already exists in target destination.")
        else:
            try:
                # shutil.move physically relocates the file
                shutil.move(str(file_path), str(target_file_path))
                success_count += 1
            except Exception as e:
                print(f"  [X] Error moving {file_path.name}: {e}")
            
    # 4. Summary
    print("\n=== Processing Summary ===")
    print(f"Successfully moved: {success_count} files.")
    print(f"Left behind (not in CSV): {missing_count} files.")
    print(f"Data is now located in: {dest_path.absolute()}")
    print("You may safely delete the original source folder IF 'Left behind' is 0.")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    RAW_WSI_DIR = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE/WSI" 
    NEW_SPLIT_DIR = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE/WSI_Split"
    CSV_FILE_PATH = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE/cleaned_split_dataset.csv"
    
    move_wsi_dataset(
        raw_dir=RAW_WSI_DIR, 
        dest_dir=NEW_SPLIT_DIR, 
        csv_path=CSV_FILE_PATH
    )