"""
AgriAI — M1.7: Dataset Manifest Generator
Consolidates audit results, split manifests, and pipeline parameters into
a comprehensive, machine-readable JSON manifest at data/manifest.json.
"""

import json
from pathlib import Path
import pandas as pd

DATA_DIR = Path("d:/Leaf/data")
REPORTS_DIR = Path("d:/Leaf/reports")
CONFIGS_DIR = Path("d:/Leaf/configs")

def generate_manifest():
    print("=" * 70)
    print("AgriAI — M1.7: Generating Machine-Readable Dataset Manifest")
    print("=" * 70)

    audit_json_path = REPORTS_DIR / "dataset_audit.json"
    if not audit_json_path.exists():
        raise FileNotFoundError(f"Audit report not found at {audit_json_path}. Run audit_dataset.py first.")

    with open(audit_json_path, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    train_csv = DATA_DIR / "train.csv"
    val_csv = DATA_DIR / "val.csv"
    test_csv = DATA_DIR / "test.csv"

    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        raise FileNotFoundError("Split CSVs not found in data/. Run split_dataset.py first.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    manifest = {
        "dataset_name": "PlantVillage Tomato Disease Subset",
        "dataset_source": "Hughes, D. P., & Salathe, M. (2015) via spMohanty/PlantVillage-Dataset",
        "license": "CC0 1.0 Universal (Public Domain Dedication)",
        "citation": "Hughes, D. P., & Salathe, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics through machine learning and crowdsourcing. arXiv:1511.08060.",
        "verified_classes_count": audit_data["num_classes"],
        "classes": list(audit_data["per_class_counts"].keys()),
        "total_images": audit_data["total_images"],
        "per_class_counts": audit_data["per_class_counts"],
        "class_imbalance_ratio": audit_data["class_imbalance"]["imbalance_ratio"],
        "splits": {
            "strategy": "Stratified random split",
            "seed": 42,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "train_manifest": "data/train.csv",
            "val_manifest": "data/val.csv",
            "test_manifest": "data/test.csv"
        },
        "preprocessing_pipeline": {
            "resolutions": {
                "development": [128, 128],
                "final": [224, 224]
            },
            "color_space": "RGB",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "standard": "ImageNet"
            },
            "augmentation_policy": {
                "train_only": True,
                "horizontal_flip": {"probability": 0.5},
                "rotation": {"degrees": 30, "probability": 0.5},
                "color_jitter": {
                    "brightness": 0.3,
                    "contrast": 0.3,
                    "saturation": 0.3,
                    "hue": 0.1
                },
                "resized_crop": {"scale": [0.7, 1.0]},
                "vertical_flip": {
                    "default": False,
                    "note": "Deliberately excluded in base pipeline to preserve biological foliar orientation; available strictly as ablation ABL-05"
                }
            },
            "eval_policy": {
                "deterministic": True,
                "augmentations": False
            }
        },
        "integrity": {
            "corrupt_files_count": audit_data["corrupt_files_count"],
            "verification_status": "VERIFIED" if audit_data["corrupt_files_count"] == 0 else "CORRUPT_FILES_PRESENT"
        }
    }

    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset manifest successfully generated at: {manifest_path}")
    print("=" * 70)
    return manifest

if __name__ == "__main__":
    generate_manifest()
