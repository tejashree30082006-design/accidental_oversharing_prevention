"""Application Configuration Settings.

Centralized configuration for database paths, API prefixes, and server settings.
"""

from pathlib import Path

# Base directory paths
# Points to the root 'Summa' folder
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Database configuration
DATABASE_DIR = BASE_DIR / "database"
DATABASE_FILE = DATABASE_DIR / "summa.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE.as_posix()}"

# API configuration
API_V1_STR = "/api/v1"
PROJECT_NAME = "Accidental Oversharing Detector"
