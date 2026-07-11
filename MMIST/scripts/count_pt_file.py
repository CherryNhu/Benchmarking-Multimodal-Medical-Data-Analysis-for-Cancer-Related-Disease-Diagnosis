from pathlib import Path

target_folder = Path('/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/Clinical/test') 

pt_files = list(target_folder.rglob("*.pt"))
count = len(pt_files)

print("So luong file .pt:", count)