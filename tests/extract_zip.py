import os
import zipfile
import logging

# Configuration
ZIP_FILE_PATH = '/root/.ipython/multilabel_cnn_proj_v2/data/Corel-5k.zip'
EXTRACT_TO_DIR = 'data/raw/Corel-5k/'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_zip(file_path, extract_to):
    """Extracts a zip file to the specified directory."""
    if not os.path.exists(file_path):
        logging.error(f"Zip file does not exist: {file_path}")
        raise FileNotFoundError(f"Zip file does not exist: {file_path}")
    
    if not os.path.exists(extract_to):
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logging.info(f"Extracted {file_path} to {extract_to}.")
        except Exception as e:
            logging.error(f"An error occurred while extracting {file_path}: {e}")
            raise
    else:
        logging.info(f"Directory {extract_to} already exists. Skipping extraction.")

if __name__ == "__main__":
    try:
        extract_zip(ZIP_FILE_PATH, EXTRACT_TO_DIR)
    except Exception as e:
        logging.error(f"An error occurred in the extraction process: {e}")
