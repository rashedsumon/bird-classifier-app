# data_loader.py
import os
import kagglehub

def load_dataset_paths():
    """
    Downloads the dataset via kagglehub and maps the train/test/validation folder directories.
    """
    print("Downloading/Updating dataset from Kaggle...")
    # Download latest version of the specified 20 species dataset
    base_path = kagglehub.dataset_download("umairshahpirzada/birds-20-species-image-classification")
    
    # Map the primary subdirectories based on Kaggle's layout
    paths = {
        "base": base_path,
        "train": os.path.join(base_path, "train"),
        "valid": os.path.join(base_path, "valid"),
        "test": os.path.join(base_path, "test")
    }
    return paths

def get_bird_classes(train_dir_path):
    """
    Extracts sorted bird species folder names to map them to deterministic labels.
    """
    if os.path.exists(train_dir_path):
        # Read sub-folders and sort alphabetically to ensure index consistency
        classes = sorted(os.listdir(train_dir_path))
        return classes
    else:
        # Fallback list representing the dataset species if offline or unzipped
        return [
            "ABBOTTS BABBLER", "ABBOTTS BOOBY", "ABYSSINIAN GROUND HORNBILL", 
            "AFRICAN CROWNED CRANE", "AFRICAN EMERALD CUCKOO", "AFRICAN FIREFINCH", 
            "AFRICAN OYSTER CATCHER", "AFRICAN PIED HORNBILL", "AFRICAN PYGMY GOOSE", 
            "ALBATROSS", "ALBERTS TOWHEE", "ALEXANDRINE PARAKEET", "ALPINE CHOUGH", 
            "ALTAMIRA YELLOWTHROAT", "AMERICAN AVOCET", "AMERICAN BITTERN", 
            "AMERICAN COOT", "AMERICAN GOLDFINCH", "AMERICAN KESTREL", "AMERICAN PIPIT"
        ]