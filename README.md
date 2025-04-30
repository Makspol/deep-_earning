# Animal Image Classification with PyTorch (Animals-10 Dataset)

This project implements a complete image classification pipeline for animal images using the **PyTorch** framework.  
It uses the popular [Animals-10 dataset](https://drive.google.com/file/d/14K-mX-Aqqk8cNdTmZHEmMVf7y4wtuSxJ/view?usp=sharing), which contains 10 animal classes.

---

## Project Stages

1. Downloading and preparing the dataset  
2. Translating class names (Italian → English)  
3. Data analysis and visualization  
4. Stratified split into `train`, `val`, and `test`  
5. Image transformation into tensors  
6. Creating `DataLoaders`  
7. Data augmentation for the training set  
8. Data quality analysis (class distribution, noise)  
9. Visualization of examples by class  
10. Building, training, and saving the model  
11. Evaluating accuracy, plotting the confusion matrix  
12. Loading new images for testing  

---

## Environment Setup

### In Google Colab:
1. Open [Colab](https://colab.research.google.com/)
2. Connect Google Drive (for saving the model):
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Upload `animals10.zip`, then unzip it:
```python
!unzip -q animals10.zip -d ./animals10
```

---

## Class Distribution Analysis

- Classes have an imbalanced number of examples (e.g., `dog` and `spider` have nearly 5000 images, while `elephant` has only ~1400)

---

## Technologies Used

- Python 3.10+
- PyTorch / Torchvision
- Scikit-learn
- Matplotlib
- PIL / NumPy

---

> This project was implemented as part of a deep learning course.
