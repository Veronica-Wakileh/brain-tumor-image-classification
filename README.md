# Comparative Study of Image Classification Using Decision Tree, Naive Bayes, and Feedforward Neural Networks

**Course:** ENCS3340 — Artificial Intelligence
**Project:** Project 2
**Semester:** 2nd Semester 2024 / 2025

---

## Project Overview

This project implements and compares three supervised machine learning models for image classification on a brain MRI dataset. Each model represents a different learning paradigm:

- **Naive Bayes** — probabilistic reasoning
- **Decision Tree** — rule-based logic
- **Multi-Layer Perceptron (MLP)** — feedforward neural network

All three models are trained and evaluated on the same dataset, and their performance is compared using standard classification metrics. The goal is to analyze the strengths and weaknesses of each approach when applied to medical image classification.

---

## Dataset Description

The dataset consists of more than **2000 labeled brain MRI images** organized into three classes:

- `glioma_tumor`
- `no_tumor`
- `pituitary_tumor`

All images are stored in standard formats (`.jpg`, `.jpeg`, `.png`) and grouped into subfolders by class.

**Preprocessing steps:**
- Resize all images to **64×64** pixels.
- Flatten each image into a 1D vector of **4096 features**.
- Encode class labels numerically using `LabelEncoder`.
- Split the data into **80% training / 20% testing**.
- Apply `StandardScaler` to normalize features before training the MLP.

---

## Project Objectives

- Prepare and preprocess an image dataset for classification.
- Implement and train Naive Bayes, Decision Tree, and MLP classifiers.
- Evaluate each model using accuracy, precision, recall, F1-score, and confusion matrix.
- Visualize the structure of the trained Decision Tree.
- Compare the models and discuss which performs best and why.

---

## Technologies and Libraries Used

- **Python 3**
- **NumPy** — numerical operations
- **OpenCV (`cv2`)** — image loading and resizing
- **Matplotlib** — decision tree visualization
- **scikit-learn** — model training, preprocessing, and evaluation

---

## Project Structure

```
project/
├── Tumors/
│   ├── glioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
├── temp.py
└── README.md
```

The `Tumors/` folder must be in the same directory as `temp.py`. Each class must be placed inside its own subfolder.

---

## How the Program Works

1. **Image loading** — The program iterates through each subfolder inside `Tumors/`, treating each subfolder name as a class label. Images are read using OpenCV.
2. **Resizing** — Every image is resized to 64×64 pixels for consistency.
3. **Flattening** — Each resized image is flattened into a 1D array of pixel values (length 4096).
4. **Label encoding** — Class names are converted to numeric labels using `LabelEncoder`.
5. **Train/test split** — `train_test_split` divides the data into 80% training and 20% testing sets, with a fixed random seed for reproducibility.
6. **Model training** — Three models are trained on the same training set.
7. **Evaluation** — Each model is evaluated on the test set, and a unified `eval_model` function prints accuracy, precision, recall, F1-score, classification report, and confusion matrix.
8. **Tree visualization** — The first three levels of the trained Decision Tree are plotted using `plot_tree`.

---

## Implemented Models

### 1. Naive Bayes Classifier

```python
from sklearn.naive_bayes import GaussianNB

bayes = GaussianNB()
bayes.fit(X_train, Y_train)
y_nb = bayes.predict(X_test)
```

A probabilistic classifier based on Bayes' Theorem with the assumption that features are conditionally independent given the class. Fast to train and serves as a baseline. Its main limitation is that pixel independence does not hold in real images.

### 2. Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion="entropy", random_state=seed)
dt.fit(X_train, Y_train)
y_dt = dt.predict(X_test)
```

A rule-based classifier that splits data using information gain (entropy). Highly interpretable since decisions can be visualized as a tree, but prone to overfitting on high-dimensional data.

### 3. MLP Classifier (Feedforward Neural Network)

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=seed)
mlp.fit(X_train_scaled, Y_train)
y_mlp = mlp.predict(X_test_scaled)
```

A feedforward neural network with two hidden layers (256 and 128 neurons). Capable of learning non-linear patterns, but requires feature scaling and more training time.

---

## Evaluation Metrics

Each model is evaluated using the following metrics:

- **Accuracy** — proportion of correctly classified samples.
- **Precision (weighted)** — quality of positive predictions per class.
- **Recall (weighted)** — ability to find all relevant samples per class.
- **F1-score (weighted)** — harmonic mean of precision and recall.
- **Classification Report** — per-class precision, recall, and F1.
- **Confusion Matrix** — detailed breakdown of correct vs. incorrect predictions per class.

---

## Results Summary

| Model              | Accuracy   | Precision  | Recall     | F1-score   |
|--------------------|------------|------------|------------|------------|
| Naive Bayes        | ~87.32%    | ~87.48%    | ~87.32%    | ~87.23%    |
| Decision Tree      | ~91.71%    | ~91.78%    | ~91.71%    | ~91.70%    |
| MLP Classifier     | **~97.07%**| **~97.14%**| **~97.07%**| **~97.05%**|

---

## Model Comparison

- **Naive Bayes** is the fastest and simplest model and works well as a baseline. However, its assumption that pixels are independent does not match the nature of image data, which limits its accuracy.
- **Decision Tree** improves over Naive Bayes by capturing non-linear patterns and offers strong interpretability through tree visualization. The downside is that it can overfit, especially on high-dimensional input.
- **MLP Classifier** clearly performs best. It learns complex, non-linear relationships across pixel values and produces the highest accuracy. The trade-off is longer training time, the need for feature scaling, and reduced interpretability compared to the Decision Tree.

Overall, **MLP** is the most reliable choice for this medical image classification task, while Naive Bayes and Decision Tree remain useful for quick baselines and interpretable analysis.

---

## How to Run the Project

### 1. Install required libraries

```bash
pip install numpy opencv-python matplotlib scikit-learn
```

### 2. Prepare the dataset

Make sure the dataset folder is named `Tumors` and is placed in the same directory as `temp.py`, with the following structure:

```
Tumors/
├── glioma_tumor/
├── no_tumor/
└── pituitary_tumor/
```

### 3. Run the program

```bash
python temp.py
```

The script will load the dataset, train the three models, print evaluation results for each, and display the Decision Tree visualization.

---

## Notes / Assumptions

- The dataset folder **must** be named `Tumors`.
- Each class **must** be placed inside its own subfolder; the subfolder name is used as the class label.
- Supported image formats: `.jpg`, `.jpeg`, `.png`.
- All images are resized to 64×64 regardless of original dimensions.
- A fixed random seed (`42`) is used to ensure reproducible train/test splits and model results.
- Feature scaling is applied **only** for the MLP model.

---

## Conclusion

This project compared three classical machine learning models on a brain MRI classification task. The Naive Bayes classifier provided a simple and fast baseline, the Decision Tree offered better accuracy and interpretability, and the MLP achieved the strongest performance by learning complex non-linear patterns in the image data. The results highlight a clear trade-off between simplicity, interpretability, and predictive power, and demonstrate that neural networks — even relatively shallow ones — are well suited for medical image classification tasks.
