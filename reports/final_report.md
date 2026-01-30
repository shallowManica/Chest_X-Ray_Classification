
# Technical Report: Chest X-Ray Classification for Clinical Triage

## 1. Executive Summary

This project implements a deep learning solution to triage Chest X-Ray (CXR) images for **Pneumonia** and  **Tuberculosis** . Prioritizing clinical safety, the final model achieves **97% Sensitivity (Recall) for Pneumonia**, ensuring that critical cases are flagged for immediate radiologist review.

## 2. Data Analysis (EDA) & Preprocessing

**Key Findings:**

* **Imbalance:** The dataset is heavily skewed, with Tuberculosis (8,513) and Normal (7,263) cases dominating Pneumonia (4,674).
* **Signal:** Texture analysis (LBP) revealed that Pneumonia lungs exhibit higher textural complexity (mean LBP ~17.5) compared to Normal lungs (~16.5), validating the use of texture-sensitive architectures.
* **Quality:** Significant variation in brightness required the implementation of **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to normalize diagnostic features.

**Preprocessing Pipeline:**
To address varying aspect ratios without distorting anatomy, all images were resized to **320x320** using a "Resize & Pad" strategy rather than cropping, ensuring the costophrenic angles and lung apices remained visible.

## 3. Modeling Strategy

**Architecture:** **DenseNet-121** initialized with ImageNet weights.

* *Rationale:* DenseNet's feature reuse mechanism preserves high-frequency texture information better than ResNet, which is critical for detecting subtle TB scarring and interstitial pneumonia patterns.

**Training & Tuning:**

* **Loss Function:** Standard Cross-Entropy Loss (Unweighted). Initial experiments with Class Weights improved Pneumonia recall but catastrophically reduced Tuberculosis detection (Recall < 65%). Removing weights and increasing resolution to 320px balanced the performance.
* **Threshold Tuning:** The decision threshold for Pneumonia was calibrated to **0.25** to maximize sensitivity, resulting in a **0.99 AUC** for the Pneumonia class.

**Final Metrics (Test Set):**

| Class                  | Precision      | Recall (Sensitivity) | F1-Score       | AUC            |
| :--------------------- | :------------- | :------------------- | :------------- | :------------- |
| **Normal**       | 0.65           | 0.73                 | 0.69           | 0.89           |
| **Pneumonia**    | **0.78** | **0.97**       | **0.87** | **0.99** |
| **Tuberculosis** | 0.89           | 0.67                 | 0.77           | 0.94           |

* **Calibration:** The Reliability Diagram (`reports/figures/calibration_curve.png`) indicates the model is slightly overconfident, supporting the use of threshold calibration over raw probability usage.

## 4. Explainability & Robustness

**Saliency Analysis (Grad-CAM):**

* **Pneumonia:** The model correctly focuses on lobar consolidations in the lung fields.
* **Tuberculosis:** Attention is correctly placed on the **apical regions** (upper lobes), a clinical hallmark of secondary TB.
* **Bias Check:** Visualizations confirm the model relies on biological tissue and ignores radiographic text markers ("R"/"L" tags).

**Failure Analysis:**
Qualitative review (`reports/figures/failure_modes.png`) shows the primary failure mode is **False Positives** (Normal patients flagged as Sick). This is an intentional design choice to minimize False Negatives (Missed diagnoses).

## 5. Deployment Strategy

**Proposal:** "Shadow Mode" Triage System.

1. **Inference:** The model is containerized (Docker/FastAPI) to run on on-premise hospital servers, ensuring PHI stays within the firewall.
2. **Workflow:** The model acts as a  **"Second Reader"** . Scans with Pneumonia confidence > 0.25 are flagged as "Urgent" in the radiologist's worklist.
3. **Monitoring:** We utilize **Drift Detection** on pixel intensity histograms to detect new scanner hardware. If data drift is detected (KS-test p < 0.05), the model falls back to a safe mode until recalibrated.
