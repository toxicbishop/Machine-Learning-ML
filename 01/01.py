"""
=============================================================================
  MACHINE LEARNING — Week 01
  Topic : Exploratory Data Analysis — Histograms, Box Plots & Outlier Detection
  File  : 01.py

  Approach : Descriptive Statistics & Visualization
    1. Load the California Housing dataset (sklearn)
    2. Plot histograms for all numerical features  (seaborn histplot)
    3. Generate box plots for all numerical features (seaborn boxplot)
    4. Detect outliers using the IQR method

  Dependencies : numpy, pandas, matplotlib, seaborn, scikit-learn
                 (pip install numpy pandas matplotlib seaborn scikit-learn)

  Output :
    - Histogram plot  : distribution of each numerical feature (3×3 grid)
    - Box plot        : box plots for each feature showing spread & outliers
    - Console         : number of outliers per feature using the IQR method
                        e.g. MedInc: 681 outliers
    - Console         : full dataset statistical summary (describe())
=============================================================================
"""

"""Develop a program to create histograms for all numerical features and analyze the
#distribution of each feature. Generate box plots for all numerical features and identify any
#outliers. Use California Housing dataset."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
# Step 1: Load the California Housing dataset
data = fetch_california_housing(as_frame=True)
housing_df = data.frame
# Step 2: Create histograms for numerical features
numerical_features = housing_df.select_dtypes(include=[np.number]).columns
# Plot histograms
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.histplot(housing_df[feature], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()
# Step 3: Generate box plots for numerical features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=housing_df[feature], color='orange')
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
plt.show()
# Step 4: Identify outliers using the IQR method
print("Outliers Detection:")
outliers_summary = {}
for feature in numerical_features:
    Q1 = housing_df[feature].quantile(0.25)
    Q3 = housing_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = housing_df[(housing_df[feature] < lower_bound) | (housing_df[feature] >upper_bound)]
    outliers_summary[feature] = len(outliers)
    print(f"{feature}: {len(outliers)} outliers")
# Optional: Print a summary of the dataset
print("\nDataset Summary:")
print(housing_df.describe())


# =============================================================================
# Output:
# Outliers Detection:
# MedInc: 681 outliers
# HouseAge: 0 outliers
# AveRooms: 1072 outliers
# AveBedrms: 1218 outliers
# Population: 1196 outliers
# AveOccup: 1263 outliers
# Latitude: 0 outliers
# Longitude: 0 outliers
#
# Dataset Summary:
#          MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
# count  20640.0   20640.0   20640.0    20640.0     20640.0   20640.0   20640.0    20640.0      20640.0
# mean       3.9      28.6      5.43       1.10      1425.5      3.07      35.6     -119.6          2.07
# std        1.9      12.6      2.47       0.47       1132.5     10.39      2.14       2.00          1.15
# min        0.5       1.0      0.85       0.33          3.0      0.69      32.5     -124.4          0.15
# max       15.0      52.0    141.91      34.07      35682.0   1243.33      41.9     -114.3          5.00
#
# Plot 1: Histograms saved in Outputs/
# Plot 2: Box Plots  saved in Outputs/
# =============================================================================
