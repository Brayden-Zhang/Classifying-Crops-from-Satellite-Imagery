
# from IPython.core.debugger import set_trace
from pathlib import Path
import requests
import zipfile
import shutil
import os
import re

import random
import argparse
from functools import partial
from pathlib import Path
from random import shuffle
from typing import Callable, Optional, List, Dict, Any

import joblib
import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.preprocessing import label_binarize
from lightning import LightningDataModule, LightningModule
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader


# @title Next, we need to download the necessary files to be used in this practical:. (Run Cell)
raster_url = "https://drive.google.com/uc?export=download&id=1CrizA11Ri3jBtlMLu-58MDkM_aELt9RQ"
crop_url = "https://drive.google.com/uc?export=download&id=1w1pvR0ESImXhgoCdy3QO6dV03vXbmvHH"
bikes_url = "https://drive.google.com/uc?export=download&id=161vAJpEnau9pXuJEi0omDmNu38cudJq1"
paris_districts_url = "https://drive.google.com/uc?export=download&id=1XyM6U-rO963zRDmtHAUx918De3qmMqzz"
dl_arrays_url = "https://drive.google.com/uc?export=download&id=1vHws-qgnzA9JOO5CCPFZadKNnsckJ2Yd"

def download_file_from_google_drive(url, directory, file_name):
    # Extract file_id from the URL
    file_id = re.search('id=([a-zA-Z0-9_-]+)', url).group(1)

    # Google Drive URL
    base_url = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(base_url, params={'id': file_id}, stream=True)

    # Check the content type
    content_type = response.headers.get('Content-Type')

    # If it's an HTML page, it might be a confirmation page
    if 'text/html' in content_type:
        confirm_token_match = re.search('confirm=([0-9A-Za-z_]+)', response.text)
        if confirm_token_match:
            confirm_token = confirm_token_match.group(1)
            response = session.get(base_url, params={'id': file_id, 'confirm': confirm_token}, stream=True)

    # Save the content to the destination
    save_response_content(response, os.path.join(directory, file_name))

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Set the path of the directory
data_dir = Path("./files")
if data_dir.exists(): shutil.rmtree(data_dir, ignore_errors=True)
data_dir.mkdir(exist_ok=True)

# Download
download_file_from_google_drive(raster_url, data_dir, "elevation.tiff")
download_file_from_google_drive(crop_url, data_dir, "df.feather")
download_file_from_google_drive(bikes_url, data_dir, "paris_bike_stations_mercator.gpkg")
download_file_from_google_drive(paris_districts_url, data_dir, "paris_districts_utm.geojson")
download_file_from_google_drive(dl_arrays_url, data_dir, "deep_learning_arrays.zip")
with zipfile.ZipFile(os.path.join(data_dir, "deep_learning_arrays.zip"), "r") as zip_ref:
    zip_ref.extractall(data_dir)