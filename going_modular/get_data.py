"""Extracts pizza steak sushi data file
"""
import os
from pathlib import Path
import requests
import zipfile

def main(url: str,
         data_path: str = "data",
         image_dir: str = "pizza_steak_sushi"):
    
    data_path = Path("data/")
    image_path = data_path/image_dir

    if image_path.is_dir():
        print(f"{image_path} directory already exists. Skipping download...")
    else:
        print(f"Creating {image_path} directory...")
        image_path.mkdir(parents=True, exist_ok=True)

    # Download data
    with open(data_path/"pizza_steak_sushi.zip", "wb") as f:
        request = requests.get(url)
        print("Downloading pizza_steak_sushi data...")
        f.write(request.content)

    # Unzip zip folder
    with zipfile.ZipFile(data_path/"pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unziping pizza_steak_sushi data...")
        zip_ref.extractall(image_path)
    
    return {"status": "Data extracted successfully"}
