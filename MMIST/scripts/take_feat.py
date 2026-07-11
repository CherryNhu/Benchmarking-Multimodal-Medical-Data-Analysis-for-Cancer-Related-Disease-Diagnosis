import shutil
from pathlib import Path

# 1. Define the target base directory
target_base_path = Path('RunBaseline/MMIST/feat/all')

# 2. Map each modality to its corresponding list of source paths
modality_mapping = {
    'Clinical': [
        Path('ExtractFeature/feature/Clinical/MMIST')
    ],
    'MRI': [
        Path('ExtractFeature/feature/MRI_CT/mmist/CPTAC/MRI'),
        Path('ExtractFeature/feature/MRI_CT/mmist/TCGA/MRI')
    ],
    'CT': [
        Path('ExtractFeature/feature/MRI_CT/mmist/TCGA/CT'),
        Path('ExtractFeature/feature/MRI_CT/mmist/CPTAC/CT')
    ],
    'WSI': [
        Path('ExtractFeature/feature/WSI/mmist/pt_files/conch15')
    ]
}

# 3. Process each modality
for modality, source_dirs in modality_mapping.items():
    # Create the target subdirectory (e.g., MMIST/feat/all/MRI)
    # parents=True creates intermediate folders if they don't exist
    # exist_ok=True prevents errors if the folder is already there
    target_modality_dir = target_base_path / modality
    target_modality_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {modality} features...")
    
    # Loop through each source directory for the current modality
    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"  [Warning] Directory not found: {src_dir}")
            continue
            
        # Find all .pt files and copy them
        # .rglob('*.pt') will search recursively if you have subfolders, 
        # use .glob('*.pt') if they are strictly in the top folder
        file_count = 0
        for pt_file in src_dir.rglob('*.pt'):
            target_file = target_modality_dir / pt_file.name
            
            # copy2 preserves metadata like creation/modification time
            shutil.copy2(pt_file, target_file) 
            file_count += 1
            
        print(f"  -> Copied {file_count} files from {src_dir.name}")

print("\nSuccess! All modalities have been organized into:", target_base_path.absolute())