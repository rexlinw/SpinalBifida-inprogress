import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image

from config import ProjectConfig

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def scan_dataset(root: Path) -> dict:
    split_counts = defaultdict(lambda: defaultdict(int))
    unreadable_files = []
    file_records = []

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

                split_counts[split][class_name] += 1

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
                        }
                    )
                except Exception as exc:
                    unreadable_files.append({"path": str(file_path), "error": str(exc)})

    hash_to_records = defaultdict(list)
    for rec in file_records:
        hash_to_records[rec["hash"]].append(rec)

    duplicates = []
    cross_split_leakage = []
    for hsh, records in hash_to_records.items():
        if len(records) <= 1:
            continue

        duplicates.append({"hash": hsh, "count": len(records), "files": records})

        splits = sorted({r["split"] for r in records})
        if len(splits) > 1:
            cross_split_leakage.append({"hash": hsh, "splits": splits, "files": records})

    summary = {
        "dataset_root": str(root),
        "split_counts": {k: dict(v) for k, v in split_counts.items()},
        "total_images": sum(sum(v.values()) for v in split_counts.values()),
        "unreadable_count": len(unreadable_files),
        "duplicate_hash_groups": len(duplicates),
        "cross_split_leakage_groups": len(cross_split_leakage),
    }

    return {
        "summary": summary,
        "unreadable_files": unreadable_files,
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
    print(f"- Report saved to: {args.out}")


if __name__ == "__main__":
    main()
