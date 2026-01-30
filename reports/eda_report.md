
# Technical Report: Data Analysis & Preprocessing Strategy

## 1. Introduction & Data Description

This report summarizes the initial exploration of the Chest X-Ray dataset. The goal was to understand the data's quality, distribution, and characteristics to design a robust preprocessing pipeline for a 3-class classification model (Normal vs. Pneumonia vs. Tuberculosis).

### 1.1 Class Distribution & Imbalance

The dataset is organized into three splits: `train`, `test`, and `val`. I counted the images in each folder and found a significant imbalance.

**Key Findings:**

* **Total Images:** The dataset is quite large, with over 20,000 images in the training set alone.
* **Imbalance:** In the training set, `Tuberculosis` (8,513) and `Normal` (7,263) are much more common than `Pneumonia` (4,674).
* **Risk:** If we don't handle this, the model might just guess "Normal" or "TB" and ignore Pneumonia.
* **Split Strategy:** The original validation split was small compared to the training set, but it seems usable. However, to be safe, I implemented a **Stratified Shuffle Split** in my code to merge `train` and `val` and re-split them (80/20) to ensure every class is fairly represented during training.

### 1.2 Image Quality & Dimensions

The images vary wildly in size. Some are high-resolution (e.g., 2000x2000), while others are much smaller (e.g., 512x512). The aspect ratios also differ; some are wide, and some are tall.

* **Artifacts:** As seen in the samples above, there are visible text markers ("R", "L") burnt into the X-rays. The model needs to learn to ignore these.
* **Exposure:** Some images are very bright (washed out), while others are dark. This confirms we need contrast normalization.

---

## 2. Exploratory Data Analysis (EDA) Findings

### 2.1 Pixel Intensity Analysis

I plotted the pixel intensity histograms to see the brightness distribution for each class.

**Pneumonia (Orange):** This class shows a higher density of pixels in the **150–200 range** compared to the others. Since higher values are "whiter," this aligns with the clinical presentation of pneumonia, which often appears as white, cloudy "infiltrates" or consolidations in the lungs.

**Tuberculosis (Green):** TB shows a unique "bump" or secondary peak around the **60–80 range** (darker greys) and another distinct peak near  **160** . This suggests a specific pattern of tissue density unique to the TB scans in this dataset.

**Normal (Blue):** The normal scans have a smoother, more distributed curve in the mid-tones and a slightly higher tail toward the bright white end (220–240), possibly indicating clearer bone definition or different exposure settings.

* **Takeaway:** The distributions for Normal, Pneumonia, and TB are actually quite similar, and we cannot only rely on simple brightness to distinguish the classes. The model must learn texture and shape.

### 2.2 Texture Analysis (Sharpness & LBP)

Since medical diagnosis relies on texture (e.g., "cloudy" lungs in pneumonia), I tested two features:

1. **Laplacian Variance:** Measures how "sharp" an image is.
2. **Local Binary Patterns (LBP):** Measures texture complexity.

* **Sharpness:** All classes have a wide range of sharpness, but `Tuberculosis` images seem slightly sharper on average than `Pneumonia`.

**Tuberculosis (Green)** shows the highest degree of variability. While its median sharpness is similar to pneumonia, it has a much wider interquartile range and a significant number of high-value outliers. This suggests the TB dataset might come from multiple sources or contains images with very high contrast/detail.

**Pneumonia (Orange)** shows the most "consistent" in terms of sharpness. The box is tighter, meaning the images in this category have more uniform quality and edge definition.

**Normal (Blue)** has the lowest median sharpness and a lower overall distribution compared to the disease classes.

* **LBP:

  **

  * **Pneumonia (Orange):** This class exhibits the  **highest median LBP value** , approximately 17.5. This suggests that pneumonia images in this dataset generally have a more complex or "busier" local texture compared to the other two classes.
  * **Normal (Green):** Normal scans sit in the middle with a median value around 16.5.
  * **Tuberculosis (Blue):** Interestingly, tuberculosis shows the **lowest median texture complexity** (roughly 15.8), but it features the most significant spread of high-value outliers reaching above 21.

  Overall, the texture complexity (LBP) is higher for Pneumonia (orange box) compared to Normal (blue box). This makes sense because pneumonia fills the lungs with fluid, creating more "rough" textures compared to clear lungs. This confirms that texture is a useful feature for the model.

---

## 3. Preprocessing & Augmentation Pipeline

Based on the findings above, I designed the following pipeline in `src/data/dataset.py`.

### 3.1 Standardization (Resize & Pad)

Since aspect ratios vary, simply resizing images to 224x224 would stretch and distort the rib cage.

* **Decision:** I implemented a `ResizePad` function. It resizes the longest side to 224 and pads the shorter side with black.
* **Benefit:** This preserves the true shape of the heart and lungs, which is critical for medical diagnosis.

### 3.2 Contrast Normalization (CLAHE)

To fix the over-exposed and under-exposed images, I used **CLAHE** (Contrast Limited Adaptive Histogram Equalization).

* **Result:** As shown above, the "Processed" image reveals much more detail in the rib cage and spine compared to the "Original."

### 3.3 Data Augmentation

To prevent overfitting and make the model robust to different scanners, I applied clinically plausible augmentations during training:

* **Random Rotations:** Up to 10 degrees (simulating patient positioning).
* **Horizontal Flips:** Standard for X-rays.
* **Avoided:** I did *not* use aggressive warping or color jitter, as these could distort the anatomy or hide disease markers.

### 3.4 Corrupt Data Handling

I wrote a script to scan for corrupt files.

* **Strategy:** My `ChestXRayDataset` class includes a `try-except` block. If an image file is broken (unreadable), it catches the error, prints a warning, and skips the file (or returns a blank tensor) to prevent the training loop from crashing.

---

## 4. Conclusion

The data is imbalanced and varies in quality, but the EDA confirms that distinct textural features exist between the classes. The proposed pipeline—using **CLAHE** for contrast, **Padding** for shape preservation, and **Stratified Splitting** for fair evaluation—is ready for the modeling phase.
