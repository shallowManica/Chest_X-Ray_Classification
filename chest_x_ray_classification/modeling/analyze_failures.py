import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.calibration import calibration_curve
from torchvision import models
import torch.nn as nn


sys.path.append(str(Path(__file__).resolve().parents[2]))
from chest_x_ray_classification.dataset import create_dataloaders, CLASSES
from chest_x_ray_classification.config import RAW_DATA_DIR, MODELS_DIR, FIGURES_DIR


MODEL_PATH = MODELS_DIR / "densenet_unweighted.pth"

def load_model(device):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model.to(device).eval()

def analyze_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    _, _, test_loader = create_dataloaders(RAW_DATA_DIR, batch_size=32)
    
    all_probs = []
    all_labels = []
    all_paths = test_loader.dataset.file_list 
    
    print("Gathering predictions for failure analysis...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = np.argmax(all_probs, axis=1)
    
    print("Generating Calibration Curve...")
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    for i, class_name in enumerate(CLASSES):
        prob_true, prob_pred = calibration_curve((all_labels == i).astype(int), all_probs[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"{class_name}")
    
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURES_DIR / "calibration_curve.png")
    print(" Saved calibration_curve.png")

    # Find "High Confidence" Errors
    print("\nVisualizing Worst Failures...")
    failures = []
    for idx in range(len(all_labels)):
        true = all_labels[idx]
        pred = preds[idx]
        conf = all_probs[idx][pred]
        
        if true != pred:
            failures.append((conf, true, pred, all_paths[idx]))
            
    # Sort by confidence (Model was WRONG but SURE about it)
    failures.sort(key=lambda x: x[0], reverse=True)
    
    # Plot top 5 failures
    plt.figure(figsize=(15, 6))
    for i in range(min(5, len(failures))):
        conf, true_idx, pred_idx, path = failures[i]
        
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(f"True: {CLASSES[true_idx]}\nPred: {CLASSES[pred_idx]}\nConf: {conf:.2f}", color='red', fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_modes.png")
    print(" Saved failure_modes.png")

if __name__ == "__main__":
    analyze_performance()