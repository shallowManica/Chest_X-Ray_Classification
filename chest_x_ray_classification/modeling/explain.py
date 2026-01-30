import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from torchvision import models
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[2]))
from chest_x_ray_classification.dataset import CLASSES
from chest_x_ray_classification.config import RAW_DATA_DIR, MODELS_DIR, FIGURES_DIR


MODEL_PATH = MODELS_DIR / "densenet_unweighted.pth"
DATA_DIR = RAW_DATA_DIR / "test"

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        # 1. Forward Pass
        output = self.model(x)
        self.model.zero_grad()
        
        # 2. Backward Pass for specific class
        target = output[0][class_idx]
        target.backward()
        
        # 3. Generate Map
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        # Weight the channels by gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Average the channels
        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        
        # ReLU (Keep only positive influence)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Error: Model file not found.")
        sys.exit()
        
    return model.to(device).eval(), device

def generate_explanations():
    model, device = load_model()
    
    # Target Layer: The last dense block of DenseNet-121
    target_layer = model.features.denseblock4.denselayer16
    cam = GradCAM(model, target_layer)
    
    # Select 1 image from each class to visualize
    sample_images = {
        'normal': 'NORMAL',        
        'pneumonia': 'PNEUMONIA',
        'tuberculosis': 'TUBERCULOSIS' 
    }
    
    plt.figure(figsize=(15, 5))
    
    for i, (class_name, folder_name) in enumerate(sample_images.items()):
        folder_path = Path(DATA_DIR) / folder_name
        
        # Pick the first valid image we find
        img_files = list(folder_path.glob("*.jp*g")) + list(folder_path.glob("*.png"))
        if not img_files:
            continue
            
        img_path = img_files[0]
        
        # Preprocess
        img_raw = cv2.imread(str(img_path))
        img_raw = cv2.resize(img_raw, (224, 224))
        
        # Prepare Tensor
        img_tensor = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        img_tensor = torch.tensor(img_tensor).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1) / 255.0
        img_tensor = img_tensor.to(device)
        
        # Generate Heatmap for the correct class index
        # Classes: 0=Normal, 1=Pneumonia, 2=TB (Check your CLASSES list order!)
        class_idx = CLASSES.index(class_name) 
        
        heatmap = cam(img_tensor, class_idx=class_idx)
        
        # Overlay
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(img_raw, 0.6, heatmap_color, 0.4, 0)
        
        # Plot
        plt.subplot(1, 3, i+1)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"Class: {class_name.upper()}")
        plt.axis('off')
        
    save_loc = FIGURES_DIR / "grad_cam_analysis.png"
    plt.savefig(save_loc)
    print(f"\n Saved Explainability Report to {save_loc}")
    plt.show()

if __name__ == "__main__":
    generate_explanations()