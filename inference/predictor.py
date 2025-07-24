import torch.nn.functional as F
from PIL import Image
import torch

def predict_image(image: Image.Image, model_info, device):
    model = model_info['model']
    transform = model_info['transform']
    class_map = model_info['class_map']

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        confidence = torch.max(probs).item()
        pred_idx = torch.argmax(probs).item()

    class_name = class_map[pred_idx]
    prob_dict = {
        class_map[i]: round(probs[i].item() * 100, 2)
        for i in range(len(probs))
    }

    return class_name, confidence, prob_dict

