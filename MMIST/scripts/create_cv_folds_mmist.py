#!/usr/bin/env python3
"""
Create Stratified Group 5-Fold splits for MMIST and (optionally) materialize
feature directories via symlinks.

Input:
  - feature_dir: path to feat-MMIST (must contain master_dataset.csv and all/)

Output (default):
  feature_dir_cv/
    fold_0/
      CT/Clinical/MRI/WSI/{train,val,test}/...
      master_dataset.csv
    fold_1/...
    ...
"""

import argparse
import os
import shutil
from typing import Dict, Iterable, Optional

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit


def _resolve_source(feature_dir: str, modality: str, file_name: str) -> Optional[str]:
    """Resolve a file or folder in feat-MMIST (prefer all/)."""
    candidates = [
        os.path.join(feature_dir, "all", modality, file_name),
        os.path.join(feature_dir, modality, "train", file_name),
        os.path.join(feature_dir, modality, "val", file_name),
        os.path.join(feature_dir, modality, "test", file_name),
    ]
    # Some entries may be stored as .pt even if file_name lacks suffix
    candidates += [c + ".pt" for c in candidates if not c.endswith(".pt")]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _safe_link_or_copy(src: str, dst: str, link_mode: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if link_mode == "symlink":
        os.symlink(src, dst)
    elif link_mode == "copy":
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link_mode: {link_mode}")


def _assign_splits(
    case_df: pd.DataFrame,
    n_splits: int,
    seed: int,
    val_fraction: float,
    val_folds: int,
) -> Dict[str, Dict[str, set]]:
    """Return {fold_idx: {'train': set, 'val': set, 'test': set}}."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    case_ids = case_df["case_id"].tolist()
    labels = case_df["label"].tolist()
    groups = case_df["case_id"].tolist()

    fold_cases: Dict[str, Dict[str, set]] = {}
    for fold_idx, (_, test_idx) in enumerate(sgkf.split(case_ids, labels, groups)):
        test_cases = set(case_df.iloc[test_idx]["case_id"].tolist())
        remaining = case_df[~case_df["case_id"].isin(test_cases)].reset_index(drop=True)

        # Build val from remaining using StratifiedGroupKFold, fallback to GroupShuffleSplit
        val_cases: set = set()
        train_cases: set = set()
        if len(remaining) >= 2:
            try:
                n_val_splits = min(val_folds, remaining["case_id"].nunique())
                n_val_splits = max(2, n_val_splits)
                sgkf_val = StratifiedGroupKFold(
                    n_splits=n_val_splits, shuffle=True, random_state=seed
                )
                train_idx, val_idx = next(
                    sgkf_val.split(
                        remaining["case_id"].tolist(),
                        remaining["label"].tolist(),
                        remaining["case_id"].tolist(),
                    )
                )
                train_cases = set(remaining.iloc[train_idx]["case_id"].tolist())
                val_cases = set(remaining.iloc[val_idx]["case_id"].tolist())
            except ValueError:
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=val_fraction, random_state=seed
                )
                train_idx, val_idx = next(
                    gss.split(
                        remaining["case_id"].tolist(),
                        remaining["label"].tolist(),
                        remaining["case_id"].tolist(),
                    )
                )
                train_cases = set(remaining.iloc[train_idx]["case_id"].tolist())
                val_cases = set(remaining.iloc[val_idx]["case_id"].tolist())
        else:
            train_cases = set(remaining["case_id"].tolist())

        fold_cases[str(fold_idx)] = {
            "train": train_cases,
            "val": val_cases,
            "test": test_cases,
        }

    return fold_cases


def _apply_split_to_master(master_df: pd.DataFrame, split_map: Dict[str, str]) -> pd.DataFrame:
    df = master_df.copy()
    df["Split"] = df["case_id"].map(split_map).fillna("train")
    return df


def _materialize_fold(
    fold_dir: str,
    feature_dir: str,
    fold_df: pd.DataFrame,
    link_mode: str,
    dry_run: bool,
) -> None:
    modalities = ["CT", "MRI", "WSI", "Clinical"]
    for mod in modalities:
        for sp in ["train", "val", "test"]:
            os.makedirs(os.path.join(fold_dir, mod, sp), exist_ok=True)

    if dry_run:
        return

    rows = (
        fold_df[["Modality", "file_name", "Split"]]
        .dropna(subset=["Modality", "file_name", "Split"])
        .drop_duplicates()
    )
    missing = 0
    for _, row in rows.iterrows():
        mod = str(row["Modality"])
        fname = str(row["file_name"])
        split = str(row["Split"])
        src = _resolve_source(feature_dir, mod, fname)
        if src is None:
            missing += 1
            continue
        dst = os.path.join(fold_dir, mod, split, fname)
        _safe_link_or_copy(src, dst, link_mode)

    if missing:
        print(f"[WARN] Missing source entries in fold: {missing}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Stratified Group 5-fold splits for MMIST features."
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Path to feat-MMIST (must contain master_dataset.csv and all/)",
    )
    parser.add_argument(
        "--master_csv",
        type=str,
        default=None,
        help="Path to master_dataset.csv (default: <feature_dir>/master_dataset.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root for folds (default: <feature_dir>_cv)",
    )
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.2,
        help="Fallback val fraction if stratified group split fails (default: 0.2)",
    )
    parser.add_argument(
        "--val_folds",
        type=int,
        default=5,
        help="Number of folds for val split inside train (default: 5)",
    )
    parser.add_argument(
        "--link_mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Materialize features as symlink or copy (default: symlink)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only write CSVs; do not create feature links",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    args = parser.parse_args()

    feature_dir = args.feature_dir.rstrip("/")
    master_csv = args.master_csv or os.path.join(feature_dir, "master_dataset.csv")
    output_dir = args.output_dir or f"{feature_dir}_cv"

    if not os.path.exists(master_csv):
        raise FileNotFoundError(f"master_dataset.csv not found: {master_csv}")

    if not args.dry_run and not os.path.isdir(os.path.join(feature_dir, "all")):
        print("[WARN] feature_dir/all not found. Will try train/val/test as source.")

    if os.path.exists(output_dir) and os.listdir(output_dir) and not args.force:
        raise RuntimeError(
            f"Output dir not empty: {output_dir}. Use --force to proceed."
        )
    os.makedirs(output_dir, exist_ok=True)

    master_df = pd.read_csv(master_csv)
    if "case_id" not in master_df.columns:
        raise ValueError("master_dataset.csv missing 'case_id' column")
    if "Modality" not in master_df.columns or "file_name" not in master_df.columns:
        raise ValueError("master_dataset.csv missing 'Modality' or 'file_name' columns")

    label_col = "vital_status_12" if "vital_status_12" in master_df.columns else "label"
    if label_col not in master_df.columns:
        raise ValueError("master_dataset.csv missing label column (vital_status_12/label)")

    case_df = (
        master_df[["case_id", label_col]]
        .drop_duplicates(subset=["case_id"])
        .rename(columns={label_col: "label"})
        .reset_index(drop=True)
    )

    fold_cases = _assign_splits(
        case_df,
        n_splits=args.n_splits,
        seed=args.seed,
        val_fraction=args.val_fraction,
        val_folds=args.val_folds,
    )

    for fold_idx, splits in fold_cases.items():
        print(f"\n=== Fold {fold_idx} ===")
        split_map = {cid: "train" for cid in case_df["case_id"].tolist()}
        for cid in splits["val"]:
            split_map[cid] = "val"
        for cid in splits["test"]:
            split_map[cid] = "test"

        fold_df = _apply_split_to_master(master_df, split_map)
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_csv = os.path.join(fold_dir, "master_dataset.csv")
        fold_df.to_csv(fold_csv, index=False)
        print(f"Wrote CSV: {fold_csv}")

        # Materialize feature folders
        _materialize_fold(
            fold_dir=fold_dir,
            feature_dir=feature_dir,
            fold_df=fold_df,
            link_mode=args.link_mode,
            dry_run=args.dry_run,
        )

        # Summary
        summary = fold_df[["Split", label_col]].value_counts().reset_index(name="count")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
