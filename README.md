# Hand Gesture Classification for Sign Language 

A Convolutional Neural Network (CNN) model to recognize American Sign Language (ASL) letters from grayscale hand gesture images. Trained on the Sign Language MNIST dataset, this model achieves over 93% test accuracy.


## Overview

This project builds a classifier to recognize 26 letters of the English alphabet in ASL from 28x28 pixel grayscale images. We use a CNN built with Keras to process the input images and output letter predictions. The model is trained and validated on labelled image data and evaluated for performance on a held-out test set.


## Features

- CNN model trained using Keras on sign language images
- Achieves 93.67% accuracy on the test set
- Includes real-time testing with unseen test images
- Visualizes training and validation accuracy over epochs
- Displays sample images to understand the dataset


## Dataset

- Source: [Kaggle – Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Shape:
  - `train.csv`: 27,455 samples  
  - `test.csv`: 7,172 samples  
- Format:
  - Each sample is a flattened 28x28 grayscale image (784 features)  
  - Labels range from `0–25`, corresponding to letters A–Z


## Usage

1. Clone this repo:
```bash
git clone htps://github.com/ramsy1195/sign-language-cnn.git
cd sign-language-cnn
```

2. Run the notebook in Google Colab or locally using Jupyter:
  - `sign_language.ipynb` (recommended)
  - Or run the script:
  ```bash
  python sign_language.py
  ```
3. Make sure the dataset CSVs `train.csv` and `test.csv` are in the same directory.

## Accuracy
- Test Accuracy: 93.67%
- Epochs Trained: 30
- Batch Size: 128

## Visual Outputs
Sample Training Images
![Sample Images](images/sample_images.png)

Training Accuracy Plot
![Training Accuracy Plot](images/accuracy_plot.png)
