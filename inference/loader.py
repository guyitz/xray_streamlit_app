import os
import torch
import yaml
import requests
from pathlib import Path
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from inference.transforms import get_transform


def download_model(url, local_path):
    """
    Download a model from URL to local path
    
    Args:
        url (str): URL to download the model from
        local_path (str): Local path to save the model
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        print(f"📥 Downloading model from: {url}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Simple progress indicator
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\r⏳ Progress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ Successfully downloaded to: {local_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading from {url}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error downloading {url}: {e}")
        return False


def get_model_path(model_cfg):
    """
    Get the model path, downloading from URL if necessary
    
    Args:
        model_cfg (dict): Model configuration from YAML
        
    Returns:
        str or None: Path to the model file, None if failed to obtain
    """
    # Check if URL is provided
    if 'url' in model_cfg and model_cfg['url']:
        url = model_cfg['url']
        
        # Generate local path from URL filename
        filename = os.path.basename(url.split('?')[0])  # Remove query parameters
        local_path = os.path.join('./models', filename)
        
        # Check if model already exists locally
        if os.path.exists(local_path):
            print(f"🔍 Model already exists locally: {local_path}")
            return local_path
        
        # Download the model
        if download_model(url, local_path):
            return local_path
        else:
            print(f"⚠️ Failed to download model from {url}")
            return None
    
    # Use local path if provided
    elif 'path' in model_cfg and model_cfg['path']:
        path = model_cfg['path']
        if os.path.exists(path):
            return path
        else:
            print(f"⚠️ Warning: local model file '{path}' not found")
            return None
    
    else:
        print("⚠️ Warning: No URL or path provided for model")
        return None


def load_model_from_config(yaml_path, device):
    """
    Load models from configuration file, downloading from URLs when necessary
    
    Args:
        yaml_path (str): Path to the YAML configuration file
        device (torch.device): Device to load models on
        
    Returns:
        list: List of model info dictionaries
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    models_info = []
    
    for model_cfg in config['models']:
        name = model_cfg['name']
        arch = model_cfg['architecture']
        input_size = tuple(model_cfg['input_size'])
        class_map = model_cfg['class_map']
        
        print(f"\n🔄 Processing model: {name}")
        
        # Get model path (download if necessary)
        model_path = get_model_path(model_cfg)
        
        if model_path is None:
            print(f"⚠️ Skipping model '{name}' - could not obtain model file")
            continue
        
        try:
            # Load model architecture
            if arch == 'efficientnet_b3':
                model = EfficientNet.from_name('efficientnet-b3')
                num_features = model._fc.in_features
                model._fc = nn.Sequential(
                    nn.Dropout(p=0.35),
                    nn.Linear(num_features, len(class_map))
                )
            elif arch == 'vgg16':
                model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
                original_last_layer_in_features = model.classifier[6].in_features
                model.classifier[6] = nn.Sequential(
                    nn.Dropout(p=0.5),  # Use the dropout rate you trained with for VGG16_BN
                    nn.Linear(original_last_layer_in_features, len(class_map))
                )
            else:
                raise ValueError(f"Unsupported architecture: {arch}")
            
            # Load model weights
            print(f"📂 Loading model weights from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model.to(device)
            
            models_info.append({
                "name": name,
                "model": model,
                "transform": get_transform(input_size),  # Ensure input_size matches (224, 224) for VGG
                "class_map": class_map,
                "input_size": input_size,
                "arch": arch
            })
            
            print(f"✅ Successfully loaded model: {name}")
            
        except Exception as e:
            print(f"❌ Error loading model '{name}': {e}")
            continue
    
    print(f"\n🎉 Successfully loaded {len(models_info)} models")
    return models_info