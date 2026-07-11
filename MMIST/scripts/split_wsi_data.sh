#!/bin/bash

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (Bạn sửa ở đây)
# ==========================================
# Thư mục Clinical ĐÃ CHIA (chứa 3 folder con: train, val, test)
CLINICAL_SPLIT_DIR="/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/Clinical"

# Thư mục WSI TỔNG (nơi chứa tất cả các file C3L-xxxx-xx.pt đang lộn xộn)
WSI_SRC_DIR="/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/all/WSI"

# Thư mục WSI ĐÍCH (nơi sẽ chứa kết quả sau khi chia)
WSI_OUT_DIR="/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/WSI"

# ==========================================
# 2. TẠO THƯ MỤC WSI ĐÍCH
# ==========================================
echo "Đang thiết lập thư mục WSI đích..."
mkdir -p "$WSI_OUT_DIR/train"
mkdir -p "$WSI_OUT_DIR/val"
mkdir -p "$WSI_OUT_DIR/test"

# ==========================================
# 3. QUÉT VÀ ĐỒNG BỘ DỮ LIỆU
# ==========================================
echo "Đang tiến hành đồng bộ WSI theo Clinical..."

# Lặp qua tất cả các file .pt trong thư mục WSI tổng
for wsi_file in "$WSI_SRC_DIR"/*.pt; do
    
    # Bỏ qua nếu thư mục trống không có file nào
    [ -f "$wsi_file" ] || continue

    # Lấy ra tên file gốc (Ví dụ: C3L-00004-26.pt)
    filename=$(basename "$wsi_file")

    # Kỹ thuật cốt lõi: Cắt từ dấu '-' cuối cùng trở đi để lấy Patient ID
    # C3L-00004-26.pt -> C3L-00004
    patient_id="${filename%-*}"

    # Lắp ghép lại thành tên file của Clinical để đối chiếu
    clinical_filename="${patient_id}.pt"

    # 4. Kiểm tra xem bệnh nhân này nằm ở tập nào bên Clinical
    if [ -f "$CLINICAL_SPLIT_DIR/train/$clinical_filename" ]; then
        cp "$wsi_file" "$WSI_OUT_DIR/train/"
        
    elif [ -f "$CLINICAL_SPLIT_DIR/val/$clinical_filename" ]; then
        cp "$wsi_file" "$WSI_OUT_DIR/val/"
        
    elif [ -f "$CLINICAL_SPLIT_DIR/test/$clinical_filename" ]; then
        cp "$wsi_file" "$WSI_OUT_DIR/test/"
        
    else
        # In ra cảnh báo nếu có file WSI nhưng không tìm thấy file Clinical tương ứng
        echo "Cảnh báo: Bệnh nhân $patient_id có file WSI nhưng không tồn tại trong Clinical (Train/Val/Test)!"
    fi

done

# for wsi_file in "$WSI_SRC_DIR"/*.pt; do
    
#     [ -f "$wsi_file" ] || continue

#     filename=$(basename "$wsi_file")

#     # KỸ THUẬT MỚI: Dùng lệnh cut để tách chuỗi
#     # TCGA-A3-3306-01A-01-TS1...pt -> TCGA-A3-3306
#     patient_id=$(echo "$filename" | cut -d'-' -f1-3)

#     clinical_filename="${patient_id}.pt"

#     # 4. Phân bổ file
#     if [ -f "$CLINICAL_SPLIT_DIR/train/$clinical_filename" ]; then
#         cp "$wsi_file" "$WSI_OUT_DIR/train/"
        
#     elif [ -f "$CLINICAL_SPLIT_DIR/val/$clinical_filename" ]; then
#         cp "$wsi_file" "$WSI_OUT_DIR/val/"
        
#     elif [ -f "$CLINICAL_SPLIT_DIR/test/$clinical_filename" ]; then
#         cp "$wsi_file" "$WSI_OUT_DIR/test/"
        
#     else
#         echo "Cảnh báo: Bệnh nhân $patient_id có file WSI nhưng không tìm thấy trong tập Clinical!"
#     fi

# done

echo "Hoàn thành! Toàn bộ file WSI đã được căn chỉnh chuẩn xác theo Clinical."