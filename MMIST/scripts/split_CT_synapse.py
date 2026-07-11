import pandas as pd
import shutil
from pathlib import Path

def restructure_medical_dataset(raw_dir: str, dest_dir: str, csv_path: str):
    """
    Flattens a nested CT directory structure and splits it into train/test 
    based on a provided CSV file.
    """
    raw_path = Path(raw_dir)
    dest_path = Path(dest_dir)
    
    # 1. Read the CSV and strictly enforce data types for identifiers
    df = pd.read_csv(csv_path, dtype={'radiology_accession_number': str})
    
    # Drop any rows where the accession number is missing (NaN)
    df = df.dropna(subset=['radiology_accession_number'])
    
    # Strip any accidental whitespace just to be safe
    df['radiology_accession_number'] = df['radiology_accession_number'].str.strip()
    
    # Create the dictionary mapping
    split_mapping = dict(zip(df['radiology_accession_number'], df['split']))
    print(split_mapping)

    # 2. Iterate through the raw CT directory
    for accession_dir in raw_path.iterdir():
        if not accession_dir.is_dir():
            continue
            
        accession_num = accession_dir.name
        
        # 3. Determine if it belongs to train or test
        if accession_num not in split_mapping:
            print(f"⚠️ Warning: Accession {accession_num} not found in CSV. Skipping.")
            continue
            
        split_type = split_mapping[accession_num]
        
        # Create target directory path: e.g., CT_Structured/train/190310
        target_dir = dest_path / split_type / accession_num
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {accession_num} -> {split_type}...")
        
        # 4. Recursively find all files inside the nested SCANS folder
        # rglob('*') searches through all subdirectories
        for file_path in accession_dir.rglob('*'):
            if file_path.is_file():
                
                # Define where the file will be copied
                new_file_path = target_dir / file_path.name
                
                # Check for naming collisions (rare with medical UIDs, but best practice)
                if new_file_path.exists():
                    print(f"  [!] File {file_path.name} already exists in target. Renaming to avoid overwrite.")
                    new_file_path = target_dir / f"{file_path.parent.name}_{file_path.name}"
                
                # Copy the file, preserving metadata
                shutil.copy2(file_path, new_file_path)

    print("\n✅ Restructuring complete!")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    # Update these paths to match your local environment
    RAW_CT_DIRECTORY = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE/CT" 
    NEW_STRUCTURED_DIRECTORY = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE/CT_Structured"
    CSV_FILE_PATH = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE/cleaned_split_dataset.csv"
    
    restructure_medical_dataset(
        raw_dir=RAW_CT_DIRECTORY, 
        dest_dir=NEW_STRUCTURED_DIRECTORY, 
        csv_path=CSV_FILE_PATH
    )