import math
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download

CLASSES = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']

HF_REPO  = "totz07/wafer-defect-classifier"
MODEL_FILE = "best_wafer_model.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, len(CLASSES))
    )
    path = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    return model

def get_defect_density(image):
    img = np.array(image.convert('L'))
    img_resized = cv2.resize(img, (256, 256))
    _, binary = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = img_resized > 10
    binary = binary * mask
    defect_pixels = np.sum(binary > 0)
    wafer_pixels  = np.sum(mask)
    defect_ratio  = defect_pixels / wafer_pixels if wafer_pixels > 0 else 0
    return round(defect_ratio, 4)

def murphy_yield(defect_ratio, die_area_cm2=0.5, wafer_area_cm2=706.86):
    num_dies = wafer_area_cm2 / die_area_cm2
    defects_per_die = defect_ratio * num_dies * die_area_cm2 / wafer_area_cm2
    AD = defects_per_die
    if AD <= 0:
        return 100.0
    if AD > 10:
        return 0.0
    yield_val = ((1 - math.exp(-AD)) / AD) ** 2
    return round(yield_val * 100, 2)

def predict(image, model, die_area_cm2=0.5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = probs.argmax().item()

    defect_type    = CLASSES[pred]
    confidence     = probs[pred].item() * 100
    all_probs      = {cls: round(probs[i].item() * 100, 2) for i, cls in enumerate(CLASSES)}
    defect_ratio   = get_defect_density(image)
    yield_pct      = murphy_yield(defect_ratio, die_area_cm2)

    if yield_pct >= 70:
        decision = "SAVE"
    elif yield_pct >= 40:
        decision = "REVIEW"
    else:
        decision = "SCRAP"

    return {
        "defect_type": defect_type,
        "confidence":  confidence,
        "all_probs":   all_probs,
        "yield_pct":   yield_pct,
        "decision":    decision,
        "defect_ratio": defect_ratio
    }
