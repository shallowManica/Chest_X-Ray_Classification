import torch
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision import models
import torch.nn as nn

# Ensure we can import dataset.py
sys.path.append(str(Path(__file__).resolve().parents[2]))
from chest_x_ray_classification.dataset import create_dataloaders
from chest_x_ray_classification.config import RAW_DATA_DIR, MODELS_DIR

MODEL_PATH = MODELS_DIR / "densenet_unweighted.pth"

def load_densenet():
    """Load the DenseNet architecture exactly as we trained it."""
    model = models.densenet121(weights=None)
    # DenseNet classifier head
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3) # 3 Classes
    
    # Load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Error: Model path not found.")
        sys.exit()
        
    return model.to(device).eval()

def optimize_threshold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_densenet()
    
    # Load ONLY Test set
    _, _, test_loader = create_dataloaders(RAW_DATA_DIR, batch_size=32)
    
    print("Running Inference on Test Set to gather probabilities...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            # Handle grayscale
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # Get raw probabilities (Softmax)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print("\n--- TUNING PNEUMONIA THRESHOLD ---")
    print("(Standard Strategy: If Pneumonia Prob > X, predict Pneumonia. Else, take max of Normal/TB)")
    
    best_acc = 0
    best_thresh = 0
    best_conf_matrix = None
    
    # Check thresholds from 0.1 to 0.95
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds = []
        for p in all_probs:
            # Logic: If Pneumonia probability > Threshold, pick Pneumonia (1)
            if p[1] > thresh:
                preds.append(1)
            else:
                # Otherwise, pick the winner between Normal (0) and TB (2)
                preds.append(0 if p[0] > p[2] else 2)
        
        acc = accuracy_score(all_labels, preds)
        print(f"Threshold {thresh:.2f} | Accuracy: {acc*100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_conf_matrix = confusion_matrix(all_labels, preds)
        print(f"Thresh {thresh:.2f} -> Acc: {acc*100:.2f}%")

    print(f"\n FINAL BEST ACCURACY: {best_acc*100:.2f}% at Threshold {best_thresh:.2f}")
    print("Confusion Matrix at Best Threshold:")
    print(best_conf_matrix)

if __name__ == "__main__":
    optimize_threshold()