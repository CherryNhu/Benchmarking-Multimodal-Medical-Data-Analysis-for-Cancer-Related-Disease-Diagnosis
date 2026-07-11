# Benchmarking Multimodal Medical Data Analysis for Cancer-Related Disease Diagnosis

This repository contains baseline scripts, experiment utilities, and lightweight data tables used for multimodal benchmarking on MMIST and SYNAPSE settings.

## Repository Scope

Included in this GitHub repository:
- Baseline scripts and utilities.
- Data processing scripts in `MMIST/scripts/`.
- Example images in `image/`.
- Lightweight tabular metadata and split files (`*.csv`) under `MMIST/feat/`.

Excluded from this repository (too large for standard GitHub source hosting):
- Extracted feature tensors (`*.pt`) in `MMIST/feat/...`.
- Model checkpoints (`*.pth`) in `MMIST/checkpoint/...`.
- Large run artifacts in `MMIST/logs/`, `MMIST/runs/`.

## Project Structure

- `convert-image.py`: image conversion/preparation utility.
- `MMIST/scripts/`: scripts for split generation, preprocessing, and run helpers.
- `MMIST/feat/`: tabular metadata and split CSV files.
- `image/`: visual examples.
- `nhulnq.txt`, `uit.txt`: local notes/documents.

## Quick Start

### 1) Clone

```bash
git clone https://github.com/CherryNhu/Benchmarking-Multimodal-Medical-Data-Analysis-for-Cancer-Related-Disease-Diagnosis.git
cd Benchmarking-Multimodal-Medical-Data-Analysis-for-Cancer-Related-Disease-Diagnosis
```

### 2) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install project-specific dependencies according to your training/evaluation codebase.

### 3) Inspect available metadata

```bash
find MMIST/feat -name "*.csv" | sort
```

## Full Data and Checkpoints

This repository intentionally does not include heavy tensors/checkpoints.
If you need full features/checkpoints for full reproducibility, provide them via external storage (institution server, Drive, Kaggle, or Zenodo) and add the download link here.

Recommended section to fill in later:
- Full feature package link: TODO
- Full checkpoint package link: TODO
- Expected folder layout after download: `MMIST/feat/...` and `MMIST/checkpoint/...`

## Reproducibility Notes

- Keep relative paths under `MMIST/` unchanged when placing downloaded full data.
- The included CSV files define sample lists and split metadata.
- Scripts in `MMIST/scripts/` are the primary entry points for preprocessing and split handling.

## Citation

If you use this repository in academic work, please cite your paper and include this repository URL in the manuscript.
