import kagglehub
import shutil
import os

# Download dataset from Kaggle
path = kagglehub.dataset_download("buraktaci/multiple-sclerosis")
print("Dataset downloaded to:", path)

# Target directory inside this project
target_dir = os.path.join("data", "raw")
os.makedirs(target_dir, exist_ok=True)

# Copy dataset into data/raw
shutil.copytree(path, target_dir, dirs_exist_ok=True)

print("Dataset moved to data/raw")