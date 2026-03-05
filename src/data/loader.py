"""
Module pour charger le dataset chest X-ray
"""
import os
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login

def setup_huggingface_auth():
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')
    
    if not token:
        raise ValueError("❌ HUGGINGFACE_TOKEN not found in .env file")
    
    login(token=token)


def load_chest_xray_dataset(dataset_name="PAR8/chest-xray-pneumonia"):    
    try:
        return load_dataset(dataset_name, token=True)
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        raise


if __name__ == "__main__":
    # Test du module
    try:
        # Authentification
        setup_huggingface_auth()
        
        # Charger le dataset
        dataset = load_chest_xray_dataset()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
