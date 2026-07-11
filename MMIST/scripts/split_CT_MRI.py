import pandas as pd
import shutil
from pathlib import Path

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN 
# ==========================================
CLINICAL_DIR = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/Clinical')

# ĐƯỜNG DẪN NGUỒN (Đã tách biệt)
SOURCE_MRI_DIR = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/all/MRI')
SOURCE_CT_DIR = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/all/CT')

# ĐƯỜNG DẪN ĐÍCH (Đã tách biệt theo ý bạn)
OUTPUT_MRI_DIR = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/MRI')
OUTPUT_CT_DIR = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/CT')

CSV_TCGA = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/CPTAC.csv')
CSV_CPTAC = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/CPTAC.csv')

COL_PATIENT = 'Patient ID'        
COL_SERIES = 'Series Instance UID'         
COL_MODALITY = 'Modality'

# ==========================================
# 2. LẬP BẢNG TRA CỨU TỪ CLINICAL
# ==========================================
print("Đang quét thư mục Clinical để xác định train/val/test...")
patient_split_map = {}

for split in ['train', 'val', 'test']:
    split_dir = CLINICAL_DIR / split
    if not split_dir.exists():
        continue
        
    for pt_file in split_dir.glob('*.pt'):
        patient_id = pt_file.stem
        patient_split_map[patient_id] = split

# ==========================================
# 3. CHUẨN BỊ THƯ MỤC OUTPUT ĐỘC LẬP
# ==========================================
for split in ['train', 'val', 'test']:
    (OUTPUT_MRI_DIR / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_CT_DIR / split).mkdir(parents=True, exist_ok=True)

# ==========================================
# 4. ĐỌC CSV VÀ ĐỊNH TUYẾN SONG SONG
# ==========================================
print("Đang đọc dữ liệu từ CSV...")
df_tcga = pd.read_csv(CSV_TCGA)
df_cptac = pd.read_csv(CSV_CPTAC)

df_combined = pd.concat([df_tcga, df_cptac], ignore_index=True)

success_count = 0
missing_count = 0

print("Đang tiến hành phân bổ file...")
for index, row in df_combined.iterrows():
    patient_id = str(row[COL_PATIENT]).strip()
    series_uid = str(row[COL_SERIES]).strip()
    modality_raw = str(row[COL_MODALITY]).strip().upper()

    # Định tuyến: Xác định LUÔN CẢ thư mục nguồn VÀ thư mục đích
    if modality_raw == 'MR':
        current_source_dir = SOURCE_MRI_DIR
        current_output_dir = OUTPUT_MRI_DIR
    elif modality_raw == 'CT':
        current_source_dir = SOURCE_CT_DIR
        current_output_dir = OUTPUT_CT_DIR
    else:
        continue 

    # Nếu bệnh nhân có mặt trong tập Clinical
    if patient_id in patient_split_map:
        target_split = patient_split_map[patient_id]
        
        # 1. Tạo chuỗi pattern chứa dấu *
        search_pattern = f"{series_uid}_Gr*.pt"
        
        # 2. Dùng .glob() để quét và tìm TẤT CẢ các file khớp với pattern
        # current_source_dir sẽ là folder CT hoặc MRI tùy vào logic ở trên
        matched_files = list(current_source_dir.glob(search_pattern))

        # 3. Nếu danh sách matched_files có chứa ít nhất 1 file
        if matched_files:
            for actual_source_path in matched_files:
                # actual_source_path lúc này là đường dẫn thật (ví dụ: ..._Gr1.pt)
                target_path = current_output_dir / target_split / actual_source_path.name
                shutil.copy2(actual_source_path, target_path)
                success_count += 1
        else:
            missing_count += 1

print(f"\n--- TỔNG KẾT ---")
print(f"Đã copy thành công: {success_count} files.")
print(f"Không tìm thấy file nguồn: {missing_count} files.")