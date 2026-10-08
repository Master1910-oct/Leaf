"""
AgriAI — M1.2: PlantVillage Tomato Dataset Acquisition Script
Downloads the PlantVillage tomato subset from the authoritative source repository:
https://github.com/spMohanty/PlantVillage-Dataset

Using sparse checkout, this script selectively downloads ONLY the raw color tomato
disease images (10 classes), avoiding unnecessary non-tomato data (~70% of the repository).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"
TARGET_DIR = Path("d:/Leaf/data/raw/plantvillage_tomato")
TEMP_CLONE_DIR = Path("d:/Leaf/data/raw/_pv_clone")

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def acquire_dataset():
    print("=" * 70)
    print("AgriAI — PlantVillage Tomato Dataset Acquisition")
    print("Authoritative Source: Hughes & Salathe (2015) via spMohanty/PlantVillage-Dataset")
    print("=" * 70)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if target directory already has all 10 classes populated
    existing_classes = [c for c in TOMATO_CLASSES if (TARGET_DIR / c).is_dir() and len(list((TARGET_DIR / c).glob("*.JPG")) + list((TARGET_DIR / c).glob("*.jpg")) + list((TARGET_DIR / c).glob("*.png"))) > 0]
    if len(existing_classes) == 10:
        total_existing = sum(len(list((TARGET_DIR / c).glob("*.*"))) for c in existing_classes)
        print(f"Dataset already acquired at {TARGET_DIR} with {len(existing_classes)} classes and {total_existing} files.")
        return

    print(f"Target directory: {TARGET_DIR.resolve()}")
    print("Initiating sparse checkout to acquire only Tomato classes...")

    if TEMP_CLONE_DIR.exists():
        shutil.rmtree(TEMP_CLONE_DIR, ignore_errors=True)

    # 1. Initialize clone with no checkout and blob filter
    cmd_init = ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", REPO_URL, str(TEMP_CLONE_DIR)]
    print(f"Running: {' '.join(cmd_init)}")
    subprocess.run(cmd_init, check=True)

    # 2. Set sparse-checkout paths for tomato classes
    print("Configuring sparse-checkout for raw/color/Tomato*...")
    subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=str(TEMP_CLONE_DIR), check=True)
    subprocess.run(["git", "sparse-checkout", "set", "raw/color"], cwd=str(TEMP_CLONE_DIR), check=True)

    # 3. Checkout files
    print("Checking out files...")
    subprocess.run(["git", "checkout"], cwd=str(TEMP_CLONE_DIR), check=True)

    # 4. Move tomato folders to TARGET_DIR
    src_color = TEMP_CLONE_DIR / "raw" / "color"
    print(f"Locating tomato folders in {src_color}...")
    moved_count = 0
    total_imgs = 0
    for item in src_color.iterdir():
        if item.is_dir() and item.name.startswith("Tomato"):
            dest = TARGET_DIR / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(item), str(dest))
            img_count = len(list(dest.glob("*.*")))
            print(f"  Moved: {item.name} ({img_count} images)")
            moved_count += 1
            total_imgs += img_count

    print(f"Successfully acquired {moved_count} classes, total {total_imgs} images.")

    # 5. Clean up temporary clone
    print("Cleaning up temporary clone directory...")
    shutil.rmtree(TEMP_CLONE_DIR, ignore_errors=True)
    print("Dataset acquisition complete.")

if __name__ == "__main__":
    acquire_dataset()
