from pathlib import Path

# Home Directory
HOME_DIR = Path(Path(__file__).resolve()).parent.parent

# Subdirectories
MODEL_DIR = Path(HOME_DIR, "model")
ASSETS_DIR = Path(HOME_DIR, "assets")
LIB_DIR = Path(HOME_DIR, "lib")
CONFIG_DIR = Path(HOME_DIR, "config")
PAGES_DIR = Path(HOME_DIR, "pages")
LOG_DIR = Path(HOME_DIR, "logs")
SCRIPTS_DIR = Path(HOME_DIR, "scripts")
CSS_DIR = Path(HOME_DIR, "css")

DISEASE_DETECTION_MODEL = Path(MODEL_DIR, "plant_disease_full_model.pth")

DISEASE_CLASSES = [
                    'Apple___Cedar_apple_rust', 
                    'Apple___healthy', 
                    'Potato___Late_blight', 
                    'Potato___healthy',
                    'Tomato__Tomato_mosaic_virus',
                    'Tomato_healthy'
                ]

SYSTEM_PROMPT = """You are a plant disease expert. 
Respond with outlining suggested treatments or remedy for the disease in the following plant:
Plant: '{plant}'
Disease: '{disease}'
"""