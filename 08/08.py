"""
=============================================================================
  MACHINE LEARNING — Week 08
  Topic : Tree-Based Learning — Decision Tree Classifier
  File  : 08.py

  Approach : Decision Tree Classification
    1. Load the Breast Cancer dataset (sklearn)
    2. Split into train / test sets  (80 / 20)
    3. Train a DecisionTreeClassifier (CART algorithm)
    4. Evaluate accuracy on the test set
    5. Predict the class for a new sample (Benign / Malignant)
    6. Visualize the full decision tree

  No external API keys required.
  Dependencies : numpy, matplotlib, scikit-learn
                 (pip install numpy matplotlib scikit-learn)

  Output :
    - Console : Model Accuracy  e.g. 93.86%
    - Console : Predicted class for test sample  e.g. Benign / Malignant
    - Plot    : Full decision tree visualization with colour-coded nodes
=============================================================================
"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
new_sample = np.array([X_test[0]])
prediction = clf.predict(new_sample)

prediction_class = "Benign" if prediction == 1 else "Malignant"
print(f"Predicted Class for the new sample: {prediction_class}")

plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Decision Tree - Breast Cancer Dataset")
plt.show()