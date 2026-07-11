from pathlib import Path
import shutil

def flatten_features():
    # 1. Định nghĩa các đường dẫn (Paths)
    base_dir = Path("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE/Clinical/Synapse")
    target_dir = Path("/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-SYNAPSE/Clinical")
    
    # Tạo thư mục đích nếu chưa tồn tại
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Tìm tất cả các file 'features.pt' trong các thư mục con
    # rglob = recursive glob (tìm đệ quy)
    for feature_file in base_dir.rglob("features.pt"):
        
        # Lấy tên thư mục cha (VD: "P-0000239")
        parent_name = feature_file.parent.name
        
        # Tạo tên file mới (VD: "P-0000239.pt")
        new_filename = f"{parent_name}.pt"
        
        # Đường dẫn đích (VD: "Clinical/P-0000239.pt")
        target_path = target_dir / new_filename
        
        # 3. Thực hiện copy (hoặc di chuyển)
        if not target_path.exists():
            print(f"Copying: {feature_file} -> {target_path}")
            # Dùng copy2 để giữ nguyên metadata của file
            shutil.copy2(feature_file, target_path) 
        else:
            print(f"Warning: {target_path} already exists. Skipping...")

if __name__ == "__main__":
    flatten_features()