"""
Configuration file for Chest X-Ray Classification project.
Uses environment variables for sensitive paths with sensible defaults.

Note: This module reads from environment variables directly using os.getenv.
If you want to use a .env file, you'll need to load it before importing this module:
    
    from dotenv import load_dotenv
    load_dotenv()
    from chest_x_ray_classification.config import RAW_DATA_DIR
    
Alternatively, you can export environment variables directly in your shell.
"""
import os
from pathlib import Path

# Project root directory (2 levels up from this file)
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Data directories - use environment variables if set, otherwise use relative paths
RAW_DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_DIR / 'data' / 'raw' / 'archive'))
PROCESSED_DATA_DIR = Path(os.getenv('PROCESSED_DATA_DIR', PROJECT_DIR / 'data' / 'processed'))

# Model directories
MODELS_DIR = Path(os.getenv('MODELS_DIR', PROJECT_DIR / 'models'))

# Reports directories  
FIGURES_DIR = Path(os.getenv('FIGURES_DIR', PROJECT_DIR / 'reports' / 'figures'))

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
