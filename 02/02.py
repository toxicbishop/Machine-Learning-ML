"""
=============================================================================
  MACHINE LEARNING — Week 02
  Topic : Feature Relationships — Correlation Matrix, Heatmap & Pair Plot
  File  : 02.py

  Approach : Correlation & Pairwise Visualization
    1. Load the California Housing dataset (sklearn)
    2. Compute the correlation matrix
    3. Visualize the correlation matrix as a heatmap  (seaborn heatmap)
    4. Create a pair plot for pairwise feature relationships (seaborn pairplot)

  Dependencies : pandas, matplotlib, seaborn, scikit-learn
                 (pip install pandas matplotlib seaborn scikit-learn)

  Output :
    - Heatmap : correlation matrix of all California Housing features
                (annotated with r values, coolwarm colour palette)
    - Pair plot : pairwise scatter plots + KDE diagonals for all features
=============================================================================
"""

"""Question: Develop a program to Compute the correlation matrix to understand the relationships between
pairs of features. Visualize the correlation matrix using a heatmap to know which variables
have strong positive/negative correlations. Create a pair plot to visualize pairwise
relationships between features. Use California Housing dataset."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
# Step 1: Load the California Housing Dataset
california_data = fetch_california_housing(as_frame=True)
data = california_data.frame
# Step 2: Compute the correlation matrix
correlation_matrix = data.corr()
# Step 3: Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of California Housing Features')
plt.show()
# Step 4: Create a pair plot to visualize pairwise relationships
sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pair Plot of California Housing Features', y=1.02)
plt.show()


"""
=============================================================================
Output:
Plot 1: Heatmap — Correlation Matrix of California Housing Features
        Strong positive  : AveRooms & AveBedrms (r ≈ 0.85)
        Strong negative  : Latitude & Longitude  (r ≈ -0.92)
        Target (MedHouseVal) correlates most with MedInc (r ≈ 0.69)

Plot 2: Pair Plot — Pairwise relationships between all features
        Diagonal shows KDE distribution of each individual feature
        Saved in Outputs/
=============================================================================
"""
