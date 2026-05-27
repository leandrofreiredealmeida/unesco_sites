import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "unesco_sites"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/cluster_tab.py",
    f"src/{project_name}/components/temporal_tab.py",
    f"src/{project_name}/components/similarity_tab.py",
    f"src/{project_name}/components/criteria_tab.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/data/loader.py",
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/clustering.py",
    f"src/{project_name}/models/similarity.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "notebooks/01_eda.ipynb",
    "notebooks/02_modeling.ipynb",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "logs/.gitkeep",
    "app.py",
    "setup.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
