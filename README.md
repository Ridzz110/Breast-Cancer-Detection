# Breast Cancer Detection using Mammography

A comprehensive machine learning project for detecting breast cancer from mammogram images using two distinct approaches: traditional feature extraction with classical ML models and deep learning with CNNs.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset Preprocessing](#dataset-preprocessing)
- [Methodology](#methodology)
  - [Method 1: Feature Extraction + Classical ML](#method-1-feature-extraction--classical-ml)
  - [Method 2: Deep Learning Approach](#method-2-deep-learning-approach)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)

## 🔍 Overview

This project implements two different approaches to classify mammogram images for breast cancer detection:
1. **Feature-based approach** using traditional machine learning algorithms
2. **Deep learning approach** using CNNs and transfer learning

Both methods address the challenge of class imbalance in medical imaging datasets and achieve competitive performance.

## 📊 Dataset Preprocessing

### Handling Class Imbalance
- Applied **oversampling** techniques to balance the distribution across train, validation, and test sets
- Ensures fair representation of both benign and malignant cases
<img width="816" height="212" alt="image" src="https://github.com/user-attachments/assets/3c7ba557-a0c8-4b77-9092-6474b4997b46" />
<img width="767" height="425" alt="image" src="https://github.com/user-attachments/assets/9790357e-789c-452a-adf0-2424515efb20" />


### Image Enhancement Pipeline
1. **Median Filtering**: Applied to reduce noise and smooth the mammogram images
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhanced contrast between low-frequency and high-frequency regions
   - Critical for highlighting tumors, which appear as the brightest and highest-frequency areas in mammograms
<img width="812" height="433" alt="image" src="https://github.com/user-attachments/assets/54d37a8f-2294-47ca-a767-e767258c46ef" />

###Region of Interest (ROI) Segmentation
Before feature extraction, ROI segmentation was performed on the enhanced mammogram images to isolate the regions most likely to contain abnormalities. This step ensures that the extracted features are computed only from relevant tissue areas, improving the discriminative power of the statistical features and reducing noise from irrelevant background regions.
<img width="801" height="431" alt="image" src="https://github.com/user-attachments/assets/62d0d0ca-21c6-4973-86c0-4e2e3ed3d52e" />


## 🔬 Methodology

### Method 1: Feature Extraction + Classical ML

#### Feature Engineering
Extracted statistical features from preprocessed mammogram images:
- **Mean**: Average intensity value
- **Standard Deviation**: Intensity variation measure
- **Variance**: Squared deviation from mean
- **Entropy**: Measure of randomness/complexity
- **Skewness**: Distribution asymmetry
- **Kurtosis**: Distribution tail heaviness

Features were saved to CSV files (`train.csv`, `test.csv`, `valid.csv`) for model training.

#### Models Trained
```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}
```

#### Results - Classical ML Models

**Logistic Regression**
```
Confusion Matrix:
[[ 95 113]
 [ 89 119]]
 
Accuracy: 51.4%
```

**SVM (RBF Kernel)**
```
Confusion Matrix:
[[ 82 126]
 [ 62 146]]
 
Accuracy: 54.8%
```

**Random Forest**
```
Confusion Matrix:
[[168  40]
 [158  50]]
 
Accuracy: 52.4%
```

**Gradient Boosting**
```
Confusion Matrix:
[[120  88]
 [113  95]]
 
Accuracy: 51.7%
```
<img width="821" height="426" alt="image" src="https://github.com/user-attachments/assets/295bff8d-46ab-4a0a-af16-55a5d3b6136e" />

### Method 2: Deep Learning Approach

#### Data Augmentation
Training data generator with augmentation parameters:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values
    rotation_range=15,           # Random rotation
    width_shift_range=0.1,       # Horizontal shift
    height_shift_range=0.1,      # Vertical shift
    shear_range=0.1,            # Shear transformation
    zoom_range=0.1,             # Random zoom
    horizontal_flip=True,        # Horizontal flip
    fill_mode='nearest'          # Fill missing pixels
)
```

#### Custom CNN Architecture

Built a CNN from scratch with the following architecture:

```
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │ Params        │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 222, 222, 100)  │ 2,800         │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 111, 111, 100)  │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 109, 109, 100)  │ 90,100        │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 54, 54, 100)    │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 52, 52, 64)     │ 57,664        │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 50, 50, 64)     │ 36,928        │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 25, 25, 64)     │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 40000)          │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 64)             │ 2,560,064     │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 64)             │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 32)             │ 2,080         │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 32)             │ 0             │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 32)             │ 1,056         │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 1)              │ 33            │
└─────────────────────────────────┴────────────────────────┴───────────────┘
Total params: 2,750,725
```

**Custom CNN Results**
```
Confusion Matrix:
[[253 195]
 [260 188]]
 
Accuracy: 49.2%
```

#### Transfer Learning Models

Fine-tuned pre-trained models on mammogram dataset:

**VGG16**
```
Confusion Matrix:
[[448   0]
 [448   0]]
 
Accuracy: 50.0%
Note: Model shows bias - classifying all samples as one class
```

**ResNet50**
```
Confusion Matrix:
[[307 141]
 [319 129]]
 
Accuracy: 48.7%
```

**EfficientNetB0**
```
Confusion Matrix:
[[359  89]
 [337 111]]
 
Accuracy: 52.5%
```

## 📈 Results Summary

| Approach | Best Model | Accuracy |
|----------|-----------|----------|
| Feature Extraction + Classical ML | SVM (RBF Kernel) | 54.8% |
| Deep Learning | EfficientNetB0 | 52.5% |

## 🛠️ Technologies Used

- **Python 3.x**
- **Image Processing**: OpenCV, scikit-image
- **Machine Learning**: scikit-learn
- **Deep Learning**: TensorFlow/Keras
- **Data Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

💻 Installation
bash# Clone the repository
git clone https://github.com/yourusername/breast-cancer-detection.git
pip install -r requirements.txt

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow opencv-python scikit-image
🚀 Usage
The entire project is contained in a single Jupyter notebook that implements both approaches:
bash# Launch Jupyter Notebook
jupyter notebook breastcancerdetectioncnn.ipynb
The notebook includes:

Data preprocessing and oversampling
Image enhancement (Median Filter + CLAHE)
Feature extraction and classical ML models (Method 1)
CNN training with data augmentation (Method 2)
Transfer learning with pre-trained models (Method 2)

Simply run all cells sequentially to reproduce the results.
📝 Future Improvements

Address class imbalance with more sophisticated techniques (SMOTE, class weights)
Hyperparameter tuning for better model performance
Ensemble methods combining multiple models
Implement cross-validation for robust evaluation
Add explainability techniques (Grad-CAM, SHAP) for model interpretability

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Dataset source: https://www.kaggle.com/datasets/hayder17/breast-cancer-detection
Inspired by research in medical image analysis and computer-aided diagnosis \
resources
https://cdn.infra.unriddle.ai/documents/u113n26gz56hj34r3w0nmw9o
https://cdn.infra.unriddle.ai/documents/tp19pd44gk5bjyicm58ini1m


Note: This is a research project for educational purposes. Medical decisions should always be made by qualified healthcare professionals.
