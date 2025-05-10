# 🧠 Breast Cancer Detection Using Machine Learning

This project aims to detect breast cancer using machine learning and deep learning techniques, focusing on early, low-cost, and accurate diagnosis from mammogram images. The objective is to classify tumors as **benign** or **malignant** using image data and structured features.

## 📌 Project Overview

Breast cancer is one of the most common cancers affecting women globally. Early detection can drastically improve survival rates. This project uses deep learning models—especially Convolutional Neural Networks (CNNs)—to automate tumor classification and assist in clinical decision-making.

## 📂 Dataset

- Source: [Kaggle - Breast Cancer Detection Dataset](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection)
- Preprocessing:
  - All images resized to `224x224`
  - Normalized pixel values
  - Augmented using Keras `ImageDataGenerator`

## 🛠️ Tech Stack

- **Languages**: Python
- **Libraries**: TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib, Seaborn
- **Models Used**:
  - Custom CNN
  - VGG16 (Fine-Tuned)
  - ResNet50
  - EfficientNetB0

## 🧪 Methodology

1. **Data Preprocessing**:
   - Normalization
   - Augmentation (rotation, zoom, flipping)
   - Split into training and validation sets

2. **Model Training**:
   - Transfer learning using VGG16, ResNet50, and EfficientNetB0
   - Binary classification (benign vs. malignant)
   - Evaluated using Accuracy, Precision, Recall, F1-score, AUC-ROC

3. **Model Evaluation**:
   - Best accuracy: **VGG16** (Test Accuracy ~62%)
   - Visualized using confusion matrix and performance graphs

## 📈 Results

| Model           | Test Accuracy | AUC-ROC   | Training Time (s) |
|----------------|---------------|-----------|--------------------|
| Custom CNN     | 61.9%         | 0.56      | 144.6              |
| VGG16          | **62.2%**     | **0.62**  | 115.3              |
| ResNet50       | 38.1%         | 0.58      | 121.6              |
| EfficientNetB0 | 42.6%         | 0.53      | 140.5              |

## 📉 Challenges

- **Data Imbalance**: More benign samples led to biased results
- **Overfitting**: Especially in ResNet50 and custom CNN
- **Limited Resources**: Deep networks required high computation

## 🔍 Improvements

- Use SMOTE or class weights to handle imbalance
- Explore ensemble or attention-based methods
- Deploy via web/mobile apps for real-world testing

## 🌐 Applications

- Clinical decision support
- Remote cancer detection via mobile imaging
- Research in AI-based medical diagnostics

## ✅ Conclusion

This project successfully demonstrates how deep learning and transfer learning can help detect breast cancer from mammogram images. VGG16 proved to be the best model in terms of performance. The project also highlights the potential for AI-driven, accessible, and low-cost diagnostic tools.

---

