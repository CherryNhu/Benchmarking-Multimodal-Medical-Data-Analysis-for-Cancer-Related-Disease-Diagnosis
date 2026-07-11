#!/bin/bash

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (Bạn sửa ở đây)
# ==========================================
SOURCE_DIR="/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/all/Clinical"

# Đường dẫn đến các thư mục đích
TRAIN_DIR="/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/Clinical/train"
VAL_DIR="/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/Clinical/val"
TEST_DIR="/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/Clinical/test"

# ==========================================
# 2. TẠO THƯ MỤC
# ==========================================
echo "Đang tạo các thư mục đích..."
mkdir -p "$TRAIN_DIR"
mkdir -p "$VAL_DIR"
mkdir -p "$TEST_DIR"

# ==========================================
# 3. LẤY DANH SÁCH FILE VÀ XÁO TRỘN
# ==========================================
echo "Đang xáo trộn danh sách file..."
# Lấy các file có đuôi .pt và dùng lệnh 'shuf' để xáo trộn ngẫu nhiên
FILES=$(ls "$SOURCE_DIR" | grep '\.pt$' | shuf)

# Đếm tổng số lượng file (đếm số dòng)
TOTAL_FILES=$(echo "$FILES" | wc -l)

if [ "$TOTAL_FILES" -eq 0 ] || [ -z "$FILES" ]; then
    echo "Lỗi: Không tìm thấy file .pt nào trong $SOURCE_DIR!"
    exit 1
fi

# ==========================================
# 4. TÍNH TOÁN RANH GIỚI CHIA TỈ LỆ (80-10-10)
# ==========================================
TRAIN_LIMIT=$((TOTAL_FILES * 80 / 100))
VAL_LIMIT=$((TOTAL_FILES * 10 / 100))
VAL_END=$((TRAIN_LIMIT + VAL_LIMIT))

echo "Tổng số file Clinical: $TOTAL_FILES"
echo "Train: $TRAIN_LIMIT | Val: $VAL_LIMIT | Test: $((TOTAL_FILES - VAL_END))"

# ==========================================
# 5. TIẾN HÀNH PHÂN BỔ (COPY)
# ==========================================
CURRENT_INDEX=0

for FILE in $FILES; do
    CURRENT_INDEX=$((CURRENT_INDEX + 1))
    
    # Xác định file này sẽ đi vào thư mục nào
    if [ "$CURRENT_INDEX" -le "$TRAIN_LIMIT" ]; then
        TARGET_DIR="$TRAIN_DIR"
    elif [ "$CURRENT_INDEX" -le "$VAL_END" ]; then
        TARGET_DIR="$VAL_DIR"
    else
        TARGET_DIR="$TEST_DIR"
    fi

    # Tiến hành copy file sang thư mục tương ứng
    # Lưu ý: Dùng 'cp' thay vì 'mv' để giữ lại tập dữ liệu gốc phòng ngừa rủi ro
    cp "$SOURCE_DIR/$FILE" "$TARGET_DIR/"
done

echo "Hoàn thành! Đã chia xong dữ liệu Clinical thành train/val/test."