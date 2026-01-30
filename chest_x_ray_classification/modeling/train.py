import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models


sys.path.append(str(Path(__file__).resolve().parents[2]))
from chest_x_ray_classification.dataset import create_dataloaders, CLASSES
from chest_x_ray_classification.config import RAW_DATA_DIR, MODELS_DIR

PARAMS = {
    "model_name": "densenet121",
    "batch_size": 16,
    "epochs": 20,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "unfreeze_layers": True,  
    "seed": 42,
    "resume_path": None
}


def build_model(num_classes=3, pretrained=True):
    """
    Loads ResNet-50 and replaces the final layer for 3-class classification.
    Rationale: ResNet-50 balances depth and complexity for medical imaging.
    """
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # We repeat the channel 3 times to fit the architecture.
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
            
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
                
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total


def main():
    # 1. Setup Device & Reproducibility 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(PARAMS["seed"])
    print(f"Using device: {device}")

    # 2. Load Data 
    train_loader, val_loader, test_loader = create_dataloaders(
        RAW_DATA_DIR, batch_size=PARAMS["batch_size"]
    )

    # 3. Handle Class Imbalance 
    # We calculate weights based on the training set distribution
    # This penalizes the model more for missing rare classes (Pneumonia)
    all_labels = []
    for _, y in train_loader.dataset:
        all_labels.append(y.item())
        
    class_weights = compute_class_weight(
        class_weight="balanced", 
        classes=np.unique(all_labels), 
        y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Computed Class Weights: {class_weights}")

    # # 4. Initialize Model, Loss, Optimizer
    model = build_model(num_classes=len(CLASSES)).to(device)
    print("Using Standard CrossEntropyLoss (No Weights) to fix TB recall...")
    criterion = nn.CrossEntropyLoss() 

    # Optimizer & Scheduler 
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"], weight_decay=PARAMS["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # 5. MLflow Tracking 
    mlflow.set_experiment("ChestXRay_Classification")
    
    with mlflow.start_run():
        mlflow.log_params(PARAMS)
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_limit = 5

        # SAFE SAVE PATH 
        save_path = MODELS_DIR / "densenet_unweighted.pth"
        
        for epoch in range(PARAMS["epochs"]):
            print(f"\nEpoch {epoch+1}/{PARAMS['epochs']}")
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Step Scheduler
            scheduler.step(val_loss)
            
            # Log Metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            
            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), str(save_path))
                print(">>> New Best Model Saved!")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_limit:
                    print("Early stopping triggered.")
                    break

if __name__ == "__main__":
    main()