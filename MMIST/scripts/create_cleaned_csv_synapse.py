import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def prepare_dataset(input_csv_path, output_csv_path):
    # 1. Load the master CSV
    print(f"Loading data from {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print("Error: The master CSV file was not found. Please check the path.")
        return

    # 2. Filter the required columns
    columns_to_keep = [
        'radiology_accession_number', 
        'dmp_pt_id', 
        'pdl1_image_id', 
        'slide_id', 
        'label'
    ]
    df_filtered = df[columns_to_keep].copy()

    # --- UPDATED CODE: Impute missing data to protect rows ---
    missing_mask = df_filtered['dmp_pt_id'].isna()
    missing_count = missing_mask.sum()
    
    if missing_count > 0:
        print(f"Warning: Found {missing_count} rows with missing 'dmp_pt_id'.")
        print("Assigning unique temporary IDs to preserve these rows...")
        
        # Create a list of unique synthetic IDs: "UNKNOWN_0", "UNKNOWN_1", etc.
        synthetic_ids = [f"UNKNOWN_{i}" for i in range(missing_count)]
        
        # Inject these synthetic IDs into the missing slots
        df_filtered.loc[missing_mask, 'dmp_pt_id'] = synthetic_ids
    # ---------------------------------------------------------

    # 3. Patient-Level Split (80/20)
    # Now it is safe to run because there are zero NaN values in 'dmp_pt_id'
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    
    df_filtered['split'] = 'unassigned'
    
    train_idx, test_idx = next(gss.split(df_filtered, groups=df_filtered['dmp_pt_id']))
    
    df_filtered.iloc[train_idx, df_filtered.columns.get_loc('split')] = 'train'
    df_filtered.iloc[test_idx, df_filtered.columns.get_loc('split')] = 'test'

    # 4. Save the new CSV
    df_filtered.to_csv(output_csv_path, index=False)
    print(f"Success! Filtered and split dataset saved to {output_csv_path}")
    
    print("\n--- Split Summary ---")
    print(df_filtered['split'].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    MASTER_CSV = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE/master_csv.csv"
    OUTPUT_CSV = "/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE/cleaned_split_dataset.csv"
    
    prepare_dataset(MASTER_CSV, OUTPUT_CSV)