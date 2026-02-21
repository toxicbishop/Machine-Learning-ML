# Machine Learning (BCSL606)

Welcome to the **BCSL606: Machine Learning Basics** repository. This project is a curated collection of lab exercises and experiments designed to understand the fundamental concepts of Machine Learning, data preprocessing, visualization, and dimensionality reduction.

---

## 📂 Repository Structure

The repository is organized into sequential folders, each representing a specific experiment or topic in the Machine Learning curriculum.

| Folder | Topic | Description |
| :--- | :--- | :--- |
| `01/` | **Data Exploration & Outlier Detection** | Analysis of the California Housing dataset, outlier detection using Interquartile Range (IQR), and visual distribution analysis. |
| `02/` | **Correlation & Pair Relationships** | Visualizing feature correlations using Heatmaps and Pair Plots to understand data dependencies. |
| `03/` | **Dimensionality Reduction (PCA)** | Implementing Principal Component Analysis on the Iris dataset to visualize high-dimensional data in 2D. |
| `04/` | **Concept Learning (Find-S)** | Implementing the Find-S algorithm to find the most specific hypothesis from a dataset. |
| `05/` | **k-Nearest Neighbors (k-NN)** | Instance-based learning using Euclidean distance to classify 1D data points. |
| `06/` | **Locally Weighted Regression** | Non-parametric regression using a Gaussian kernel to fit a noisy sine wave. |
| `07/` | **Linear & Polynomial Regression** | Linear and Polynomial Regression using California Housing and Auto MPG datasets. |
| `08/` | **Decision Tree Classifier** | Tree-based classification on the Breast Cancer dataset using the CART algorithm. |
| `09/` | **Naïve Bayes Classifier** | Probabilistic classification using Gaussian Naïve Bayes on the Olivetti Faces dataset. |
| `10/` | **K-Means Clustering** | Unsupervised learning and PCA visualization on the Breast Cancer dataset. |

---

## 🚀 Getting Started

To run these experiments locally, follow the steps below.

### 📋 Prerequisites

Ensure you have Python installed. It is recommended to create a virtual environment before installing the required libraries.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment (Windows)
.\venv\Scripts\activate

# Activate the virtual environment (macOS/Linux)
source venv/bin/activate
```

Install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 🏃 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/toxicbishop/Machine-Learning-ML.git
   ```

2. Navigate to the specific experiment folder:

   ```bash
   cd 01
   ```

3. Run the Python script:

   ```bash
   python 01.py
   ```

---

## 📊 Experiments Overview

### 1. Data Exploration & Outliers (`01/`)

- **Dataset**: California Housing
- **Techniques**: IQR for outlier detection, Histograms, Box Plots.
- **Key Learning**: Understanding data distribution and identifying noise.

### 2. Feature Relationships (`02/`)

- **Dataset**: California Housing
- **Techniques**: Correlation Matrix (Heatmap), Seaborn PairPlot.
- **Key Learning**: Identifying multi-collinearity and feature importance.

### 3. Principal Component Analysis (`03/`)

- **Dataset**: Iris Dataset
- **Techniques**: PCA (Scikit-Learn).
- **Key Learning**: Reducing feature space while preserving variance.

### 4. Concept Learning (Find-S) (`04/`)

- **Dataset**: Custom Training Data (CSV)
- **Techniques**: Find-S Algorithm.
- **Key Learning**: Finding the most specific hypothesis from positive training examples.

### 5. k-Nearest Neighbors (`05/`)

- **Dataset**: Randomly generated 1D data points
- **Techniques**: k-NN Classification, Euclidean distance.
- **Key Learning**: Instance-based learning and the effect of 'k' values on classification boundaries.

### 6. Locally Weighted Regression (`06/`)

- **Dataset**: Synthesized noisy sine wave
- **Techniques**: Locally Weighted Regression (LWR), Gaussian kernel.
- **Key Learning**: Non-parametric regression techniques for complex, non-linear relationships.

### 7. Linear & Polynomial Regression (`07/`)

- **Dataset**: California Housing & Auto MPG
- **Techniques**: Linear Regression, PolynomialFeatures, MSE, R² Score.
- **Key Learning**: Modeling linear and polynomial relationships between continuous variables.

### 8. Decision Tree Classifier (`08/`)

- **Dataset**: Breast Cancer
- **Techniques**: Decision Tree (CART algorithm), Gini impurity, Tree visualization.
- **Key Learning**: Building interpretable tree-based models for classification tasks.

### 9. Naïve Bayes Classifier (`09/`)

- **Dataset**: Olivetti Faces
- **Techniques**: Gaussian Naïve Bayes, cross-validation, confusion matrix.
- **Key Learning**: Applying probabilistic generative models to predictive classification.

### 10. K-Means Clustering (`10/`)

- **Dataset**: Breast Cancer
- **Techniques**: K-Means Clustering, Principal Component Analysis (PCA).
- **Key Learning**: Unsupervised clustering and visualizing high-dimensional clusters in 2D space.

---
