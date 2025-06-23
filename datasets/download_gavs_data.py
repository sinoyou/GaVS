import os
import zipfile
from huggingface_hub import snapshot_download
from pathlib import Path

# 1. Download the dataset repository from Hugging Face Hub
repo_id = "sinoyou/gavs-data"
local_dir = "./gavs-data"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # copy files instead of symlinking
)

# 2. Unzip all .zip files in the "dataset" folder
dataset_zip_dir = Path(local_dir) / "dataset"
for zip_path in dataset_zip_dir.glob("*.zip"):
    print(f"Unzipping: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_path.parent)

# 3. Unzip all .zip files in the "result_and_comparison" folder
result_zip_dir = Path(local_dir) / "result_and_comparison"
for zip_path in result_zip_dir.glob("*.zip"):
    print(f"Unzipping: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(zip_path.parent)
