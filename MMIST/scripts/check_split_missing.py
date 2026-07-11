import csv
from pathlib import Path
from typing import Set

def extract_uids_from_csv(csv_paths: list[str], target_column: str = 'Series Instance UID') -> Set[str]:
    """Reads multiple CSVs and returns a set of unique UIDs from the target column."""
    uids = set()
    for path in csv_paths:
        try:
            with open(path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if target_column in row and row[target_column]:
                        uids.add(row[target_column].strip())
        except FileNotFoundError:
            print(f"Warning: CSV file not found at {path}")
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    return uids

def process_dicom_files(all_folder: str, split_folder: str, csv_files: list[str], output_file: str):
    # 1. Setup paths using pathlib
    path_all = Path(all_folder)
    path_split = Path(split_folder)

    # 2. Get filenames as sets for O(1) difference calculation
    # We only take the name of the file, ignoring directories
    all_files = {f.name for f in path_all.rglob('*') if f.is_file()}
    split_files = {f.name for f in path_split.rglob('*') if f.is_file()}

    # 3. Find files in 'all' but NOT in 'split'
    missing_files = all_files - split_files

    # 4. Load UIDs from both CSV files into a fast-lookup set
    valid_uids = extract_uids_from_csv(csv_files)

    found_in_csv = []
    not_found_in_csv = []

    # 5. Check each missing file against the CSV data
    for filename in missing_files:
        # Cut the string at the first '_' and take the first part [0]
        uid_from_filename = filename.split('_')[0]

        if uid_from_filename in valid_uids:
            found_in_csv.append(filename)
        else:
            not_found_in_csv.append(filename)

    # 6. Write results to the output text file
    with open(output_file, mode='w', encoding='utf-8') as out_f:
        out_f.write("=== FILES FOUND IN CSV ===\n")
        if found_in_csv:
            for f in found_in_csv:
                out_f.write(f"{f}\n")
        else:
            out_f.write("None\n")

        out_f.write("\n=== FILES NOT FOUND IN CSV ===\n")
        if not_found_in_csv:
            for f in not_found_in_csv:
                out_f.write(f"{f}\n")
        else:
            out_f.write("None\n")
            
    print(f"Process completed successfully. Check the results in {output_file}")

# --- How to run the code ---
if __name__ == "__main__":
    # Update these paths to match your actual directories and files
    ALL_DIR = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/all/CT"
    SPLIT_DIR = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/CT" 
    CSV_LIST = ["/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/TCIA_TCGA-KIRC_09-16-2015-nbia-digest(Metadata) (1).csv",
                 "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/TCIA-CPTAC-CCRCC_v11_20230818-nbia-digest(Metadata) (1).csv"]
    OUTPUT_TXT = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/logs/CT_unsplited.txt"

    process_dicom_files(ALL_DIR, SPLIT_DIR, CSV_LIST, OUTPUT_TXT)