CUDA_VISIBILE_DEVICES=2,3 python /mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/mmist-ccRCC-MMIST/src/main.py --stage ablation --ablation_mode all --feature_dir /mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-MMIST --clinical_file /mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/feat/feat-MMIST/master_dataset.csv

# watch -n 1 nvidia-smi
# python -m tensorboard.main --logdir=/mmlab_students/storageStudents/nguyenvd/UIT2024_medicare/RunBaseline/MMIST/runs