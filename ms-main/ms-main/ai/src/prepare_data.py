import os
import random
import shutil

random.seed(42)

RAW = "data/raw"
OUT = "data/processed_clean"

mapping = {
    "ms": ["MS Axial_crop", "MS Saggital_crop"],
    "control": ["Control Axial_crop", "Control Saggital_crop"],
}

splits = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

for label, folders in mapping.items():

    images = []

        # Collect images
    for folder in folders:
        path = os.path.join(RAW, folder)

        if not os.path.exists(path):
            print(f"Folder not found: {path}")
            continue

        for f in os.listdir(path):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(path, f))

    print(f"{label}: Found {len(images)} images")

    if len(images) == 0:
        continue

    random.shuffle(images)

    total = len(images)
    train_end = int(total * splits["train"])
    val_end = train_end + int(total * splits["val"])

    split_data = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_data.items():
        target_dir = os.path.join(OUT, split, label)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            shutil.copy(file, target_dir)

print("Dataset prepared successfully (image-level split)")
