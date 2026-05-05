import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Main function
if __name__ == "__main__":

    dataset_path = "Tumors"
    img_dim = (64, 64)
    seed = 42

    X = []  # Features
    Y = []  # Labels (class names)
    
    # Load and preprocess images
    for cls_name in sorted(os.listdir(dataset_path)):
        cls_path = os.path.join(dataset_path, cls_name)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(os.path.join(cls_path, fname))
                img = cv2.resize(img, img_dim)
                X.append(img.flatten())
                Y.append(cls_name)

    X = np.array(X)
    Y = np.array(Y)
    print("Loaded:", X.shape[0], "samples")

    # Encode labels to numeric
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y)

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_encoded, test_size=0.2, random_state=seed
    )

    # Model evaluation function
    def eval_model(name, y_true, y_pred):
        print(f"\n--- {name} ---")
        y_true_str = label_encoder.inverse_transform(y_true)
        y_pred_str = label_encoder.inverse_transform(y_pred)
        print("Accuracy :", accuracy_score(y_true_str, y_pred_str))
        print("Precision:", precision_score(y_true_str, y_pred_str, average="weighted", zero_division=0))
        print("Recall   :", recall_score(y_true_str, y_pred_str, average="weighted", zero_division=0))
        print("F1-score :", f1_score(y_true_str, y_pred_str, average="weighted", zero_division=0))
        print("\nClassification Report:\n", classification_report(y_true_str, y_pred_str, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_true_str, y_pred_str))

    # Naive Bayes Classifier
    bayes = GaussianNB()
    bayes.fit(X_train, Y_train)
    y_nb = bayes.predict(X_test)
    eval_model("Naive Bayes Classifier", Y_test, y_nb)

    # Decision Tree Classifier
    dt = DecisionTreeClassifier(criterion="entropy", random_state=seed)
    dt.fit(X_train, Y_train)
    y_dt = dt.predict(X_test)
    eval_model("Decision Tree Classifier", Y_test, y_dt)

    # Plot decision tree
    plt.figure(figsize=(18, 8))
    plot_tree(
        dt,
        filled=True,
        class_names=label_encoder.classes_,
        feature_names=None,
        rounded=True,
        fontsize=8,
        max_depth=3
    )
    plt.title("Visualization of the Decision Tree (First 3 Levels)")
    plt.show()

    # MLP Classifier
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=seed)
    mlp.fit(X_train_scaled, Y_train)
    y_mlp = mlp.predict(X_test_scaled)
    eval_model("MLP Classifier", Y_test, y_mlp)