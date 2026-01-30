import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torchvision import models
import torch.nn as nn

# Setup path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from chest_x_ray_classification.dataset import create_dataloaders, CLASSES
from chest_x_ray_classification.config import RAW_DATA_DIR, MODELS_DIR, FIGURES_DIR

MODEL_PATH = MODELS_DIR / "densenet_unweighted.pth"

def load_densenet(num_classes=3):
    """Rebuilds the DenseNet architecture to load weights."""
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # 1. Load Data 
    _, _, test_loader = create_dataloaders(RAW_DATA_DIR, batch_size=32)

    # 2. Load Model
    model = load_densenet(len(CLASSES))
    
    if Path(MODEL_PATH).exists():
        print(f"Loading weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f" Error: Model file not found at {MODEL_PATH}")
        return

    model.to(device)
    model.eval()

    # 3. Inference Loop
    all_preds = []
    all_labels = []
    all_probs = [] 

    print("Running inference on Test Set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            # Handle grayscale channel expansion
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
                
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # 4. Metrics & Reports
    print("\n" + "="*30)
    print("FINAL TEST RESULTS (Task 2.5)")
    print("="*30)
    
    # Text Report
    report = classification_report(all_labels, all_preds, target_names=CLASSES)
    print(report)
    
    # Save Report to file
    with open(FIGURES_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    # 5. Visualizations
    
    # A. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix (DenseNet-121)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png")
    print(f"Saved Confusion Matrix to {FIGURES_DIR}")

    # B. ROC Curve Plot (One vs Rest)
    plt.figure(figsize=(10, 8))
    all_probs = np.array(all_probs)
    
    for i, class_name in enumerate(CLASSES):
        # Create binary labels for this class vs all others
        binary_labels = (np.array(all_labels) == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2) 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(FIGURES_DIR / "roc_curve.png")
    print("Saved ROC Curve plot.")

if __name__ == "__main__":
    evaluate()