#Program1
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data=fetch_california_housing(as_frame=True)

housing_df=data.frame

numerical_features=housing_df.select_dtypes(include=[np.number]).columns

plt.figure(figsize=(15,10))
for i,feature in enumerate(numerical_features):
    plt.subplot(3,3,i+1)
    sns.histplot(housing_df[feature],kde=True,bins=30,color='blue')
    plt.title(f'Distribution of{feature}')
plt.tight_layout()

plt.show()

plt.figure(figsize=(15,10))

for i,feature in enumerate(numerical_features):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=housing_df[feature],color='orange')
    plt.title(f'Box Plot of{feature}')

plt.tight_layout()
plt.show()

print("Outliers Detection:")
outliers_summary={}
for feature in numerical_features:
    Q1=housing_df[feature].quantile(0.25)
    Q3=housing_df[feature].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers=housing_df[(housing_df[feature]<lower_bound)|(housing_df[feature]>upper_bound)]
    outliers_summary[feature]=len(outliers)
    print(f"{feature}:{len(outliers)}outliers")
    print("\nDataset Summary:")
    print(housing_df.describe())