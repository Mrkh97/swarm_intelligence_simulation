"""
Utility fonksiyonlar
Config loading, logging, vb.
"""

import yaml
import numpy as np
import torch
import random
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """YAML config dosyasını yükle"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Reproducibility için seed ayarla"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(config: Dict[str, Any]):
    """Gerekli dizinleri oluştur"""
    Path("trained_models").mkdir(exist_ok=True)
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/videos").mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """En iyi cihazı seç"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
