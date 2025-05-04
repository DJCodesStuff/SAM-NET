import os
from dotenv import dotenv_values
from pathlib import Path

# Load .env values BEFORE importing kaggle
env_path = Path('.') / '.env'
os.environ.update(dotenv_values(dotenv_path=env_path))

from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate
api = KaggleApi()
api.authenticate()

# Download dataset
DATASET = "clivelewis14/sam-net-hpc"
DOWNLOAD_DIR = "data"

api.dataset_download_files(DATASET,
                           path=DOWNLOAD_DIR,
                           unzip=True,
                           force=True
)
print(f"✅ Downloaded {DATASET} to '{DOWNLOAD_DIR}/'")
