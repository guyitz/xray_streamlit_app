import os
import torch
import yaml
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

from inference.transforms import get_transform

def load_model_from_config(yaml_path, device):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    models_info = []
    for model_cfg in config['models']:
        name = model_cfg['name']
        path = model_cfg['path']
        arch = model_cfg['architecture']
        input_size = tuple(model_cfg['input_size'])
        class_map = model_cfg['class_map']

        if not os.path.exists(path):
            print(f"⚠️ Warning: model file '{path}' not found. Skipping...")
            continue

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
                nn.Dropout(p=0.5), # Use the dropout rate you trained with for VGG16_BN
                nn.Linear(original_last_layer_in_features, len(class_map))
            )


        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        model.to(device)

        models_info.append({
            "name": name,
            "model": model,
            "transform": get_transform(input_size), # Ensure input_size matches (224, 224) for VGG
            "class_map": class_map,
            "input_size": input_size,
            "arch": arch
        })

    return models_info