import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from config import ProjectConfig

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Filename prefixes / substrings that identify image provenance
_SYNTHETIC_PREFIX = "lemon_v2_"
_REAL_SB_PREFIX = "pmc"
_ANNOTATION_MARKER = "_Annotation"


def sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def classify_provenance(filename: str) -> str:
    """Return one of 'synthetic_lemon', 'real_sb', 'annotation_overlay', or 'normal'."""
    name = os.path.basename(filename)
    if _ANNOTATION_MARKER in name:
        return "annotation_overlay"
    if name.startswith(_SYNTHETIC_PREFIX):
        return "synthetic_lemon"
    if name.lower().startswith(_REAL_SB_PREFIX):
        return "real_sb"
    return "normal"


def compute_image_stats(file_path: Path) -> dict:
    """Return basic image statistics useful for publication validation."""
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            mode = img.mode
            arr = np.array(img.convert("L"), dtype=np.float32)
        return {
            "width": width,
            "height": height,
            "mode": mode,
            "mean_brightness": float(np.mean(arr)),
            "std_brightness": float(np.std(arr)),
        }
    except Exception:
        return {}


def scan_dataset(root: Path) -> dict:
    split_counts: dict = defaultdict(lambda: defaultdict(int))
    provenance_counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    annotation_files: list = []
    unreadable_files: list = []
    file_records: list = []

    # Per-split image dimension / brightness accumulators
    dim_accum: dict = defaultdict(lambda: defaultdict(list))

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name

            for file_path in sorted(class_dir.rglob("*")):
                if not file_path.is_file() or not is_image_file(file_path):
                    continue

                fname = file_path.name
                provenance = classify_provenance(fname)

                split_counts[split][class_name] += 1
                provenance_counts[split][class_name][provenance] += 1

                if provenance == "annotation_overlay":
                    annotation_files.append(
                        {"split": split, "class": class_name, "path": str(file_path)}
                    )

                stats = compute_image_stats(file_path)
                if stats:
                    key = f"{split}/{class_name}"
                    dim_accum[key]["widths"].append(stats["width"])
                    dim_accum[key]["heights"].append(stats["height"])
                    dim_accum[key]["brightness"].append(stats["mean_brightness"])

                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    img_hash = sha256_file(file_path)
                    file_records.append(
                        {
                            "split": split,
                            "class": class_name,
                            "path": str(file_path),
                            "hash": img_hash,
                            "provenance": provenance,
                        }
                    )
                except Exception as exc:
                    unreadable_files.append({"path": str(file_path), "error": str(exc)})

    # Duplicate / leakage detection
    hash_to_records: dict = defaultdict(list)
    for rec in file_records:
        hash_to_records[rec["hash"]].append(rec)

    duplicates: list = []
    cross_split_leakage: list = []
    for hsh, records in hash_to_records.items():
        if len(records) <= 1:
            continue
        duplicates.append({"hash": hsh, "count": len(records), "files": records})
        splits = sorted({r["split"] for r in records})
        if len(splits) > 1:
            cross_split_leakage.append({"hash": hsh, "splits": splits, "files": records})

    # Aggregate image statistics per split/class
    image_stats_summary: dict = {}
    for key, data in dim_accum.items():
        widths = data["widths"]
        heights = data["heights"]
        brightness = data["brightness"]
        image_stats_summary[key] = {
            "count": len(widths),
            "width": {"min": int(min(widths)), "max": int(max(widths)), "mean": round(float(np.mean(widths)), 1)},
            "height": {"min": int(min(heights)), "max": int(max(heights)), "mean": round(float(np.mean(heights)), 1)},
            "mean_brightness": {"min": round(float(min(brightness)), 2), "max": round(float(max(brightness)), 2), "mean": round(float(np.mean(brightness)), 2)},
        }

    # Class-balance warnings
    balance_warnings: list = []
    for split, class_dict in split_counts.items():
        counts = list(class_dict.values())
        if len(counts) < 2:
            continue
        ratio = max(counts) / max(min(counts), 1)
        if ratio > 1.5:
            balance_warnings.append(
                {
                    "split": split,
                    "counts": dict(class_dict),
                    "imbalance_ratio": round(ratio, 3),
                    "warning": "Class imbalance exceeds 1.5:1 — report class weights or resampling in the paper.",
                }
            )

    # Clinical suitability flags
    clinical_flags: list = []
    total_annotation = len(annotation_files)
    if total_annotation > 0:
        clinical_flags.append(
            {
                "severity": "HIGH",
                "flag": "annotation_overlay_contamination",
                "detail": (
                    f"{total_annotation} file(s) contain '_Annotation' in their name. "
                    "These are HC18 measurement overlays (white ellipses drawn on the skull) "
                    "that do not represent clinical spinal-bifida features. "
                    "Remove them from the dataset before training; fix prepare_dataset.py "
                    "to filter filenames containing '_Annotation'."
                ),
            }
        )

    # Check synthetic ratio in training split
    train_sb = provenance_counts.get("train", {}).get("Spina_Bifida", {})
    total_train_sb = sum(train_sb.values())
    if total_train_sb > 0:
        syn_count = train_sb.get("synthetic_lemon", 0) + train_sb.get("annotation_overlay", 0)
        syn_ratio = syn_count / total_train_sb
        if syn_ratio > 0.80:
            clinical_flags.append(
                {
                    "severity": "MEDIUM",
                    "flag": "high_synthetic_ratio_in_training",
                    "detail": (
                        f"{syn_ratio * 100:.1f}% of train/Spina_Bifida images are synthetic "
                        "(lemon_v2_* images created by warping HC18 head-circumference scans). "
                        "For publication, clearly state the synthetic augmentation methodology, "
                        "validate that the synthetic lemon-sign transformation is clinically realistic, "
                        "and consider recruiting additional real fetal ultrasound cases."
                    ),
                }
            )

    test_sb = split_counts.get("test", {}).get("Spina_Bifida", 0)
    test_normal = split_counts.get("test", {}).get("Normal", 0)
    if test_sb > 0 and test_normal > 0:
        test_ratio = test_normal / test_sb
        if test_ratio > 2.0:
            clinical_flags.append(
                {
                    "severity": "MEDIUM",
                    "flag": "test_set_class_imbalance",
                    "detail": (
                        f"Test set contains {test_normal} Normal vs {test_sb} Spina_Bifida images "
                        f"({test_ratio:.1f}:1 ratio). Report per-class metrics (sensitivity, "
                        "specificity, AUC) rather than accuracy alone."
                    ),
                }
            )

    # Check whether HC18 source images (Normal class) have annotation counterparts
    # that could have leaked into Normal training data
    has_annotation_in_normal = any(
        r for r in file_records if r["class"] == "Normal" and r["provenance"] == "annotation_overlay"
    )
    if has_annotation_in_normal:
        clinical_flags.append(
            {
                "severity": "HIGH",
                "flag": "annotation_overlay_in_normal_class",
                "detail": (
                    "Annotation overlay images found in the Normal class. "
                    "These must be removed to prevent the model from learning to detect "
                    "annotation artefacts instead of pathological features."
                ),
            }
        )

    # Check for brightness domain gap between Normal and Spina_Bifida in the test split
    test_normal_key = "test/Normal"
    test_sb_key = "test/Spina_Bifida"
    if test_normal_key in dim_accum and test_sb_key in dim_accum:
        normal_bright = float(np.mean(dim_accum[test_normal_key]["brightness"]))
        sb_bright = float(np.mean(dim_accum[test_sb_key]["brightness"]))
        if sb_bright > 0 and (sb_bright / max(normal_bright, 1.0)) > 1.5:
            clinical_flags.append(
                {
                    "severity": "HIGH",
                    "flag": "brightness_domain_gap",
                    "detail": (
                        f"Mean pixel brightness in test/Spina_Bifida ({sb_bright:.1f}) is "
                        f"{sb_bright / max(normal_bright, 1.0):.1f}× higher than test/Normal ({normal_bright:.1f}). "
                        "This indicates a domain gap: Normal images are dark raw ultrasound scans "
                        "(HC18 dataset) while Spina_Bifida test panels include bright backgrounds, "
                        "labels, and colour bars from paper figures. "
                        "The model may learn brightness/background instead of pathology. "
                        "Normalise brightness or use centre-crop to exclude non-ultrasound regions."
                    ),
                }
            )

    summary = {
        "dataset_root": str(root),
        "split_counts": {k: dict(v) for k, v in split_counts.items()},
        "total_images": sum(sum(v.values()) for v in split_counts.values()),
        "unreadable_count": len(unreadable_files),
        "duplicate_hash_groups": len(duplicates),
        "cross_split_leakage_groups": len(cross_split_leakage),
        "annotation_overlay_count": total_annotation,
        "provenance_counts": {
            split: {cls: dict(prov) for cls, prov in class_dict.items()}
            for split, class_dict in provenance_counts.items()
        },
        "balance_warnings": balance_warnings,
        "clinical_flags": clinical_flags,
    }

    return {
        "summary": summary,
        "image_stats": image_stats_summary,
        "unreadable_files": unreadable_files,
        "annotation_files": annotation_files,
        "duplicates": duplicates,
        "cross_split_leakage": cross_split_leakage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset integrity and split leakage.")
    parser.add_argument(
        "--dataset",
        default=ProjectConfig().dataset_path,
        help="Path to prepared dataset root (default: dataset_prepared).",
    )
    parser.add_argument(
        "--out",
        default="dataset_report.json",
        help="Output JSON report path (default: dataset_report.json).",
    )
    args = parser.parse_args()

    report = scan_dataset(Path(args.dataset))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary = report["summary"]
    print("Dataset Validation Summary")
    print(f"- Dataset root: {summary['dataset_root']}")
    print(f"- Total images: {summary['total_images']}")
    print(f"- Unreadable files: {summary['unreadable_count']}")
    print(f"- Duplicate groups: {summary['duplicate_hash_groups']}")
    print(f"- Cross-split leakage groups: {summary['cross_split_leakage_groups']}")
    print(f"- Annotation overlay files: {summary['annotation_overlay_count']}")

    if summary["balance_warnings"]:
        print("\nClass-balance warnings:")
        for w in summary["balance_warnings"]:
            print(f"  [{w['split']}] ratio={w['imbalance_ratio']} counts={w['counts']}")

    if summary["clinical_flags"]:
        print("\nClinical / publication suitability flags:")
        for flag in summary["clinical_flags"]:
            print(f"  [{flag['severity']}] {flag['flag']}")
            print(f"    {flag['detail']}")

    print(f"\n- Report saved to: {args.out}")


if __name__ == "__main__":
    main()
