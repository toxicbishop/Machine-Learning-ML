"""
=============================================================================
  MACHINE LEARNING — Week 04
  Topic : Concept Learning — Find-S Algorithm
  File  : 04.py

  Approach : Find-S Algorithm (Most Specific Hypothesis)
    1. Read training examples from a CSV file
       (last column must be the label: 'Yes' / 'No')
    2. Initialise hypothesis h to '?' for all attributes
    3. For every POSITIVE example:
         For each attribute a_i:
           if h[i] == '?' or h[i] == a_i  → set h[i] = a_i
           else                            → generalise h[i] = '?'
    4. Output the final hypothesis h

  No external API keys required.
  Dependencies : pandas  (pip install pandas)

  Output :
    - Console : full training data table
    - Console : final hypothesis list, e.g.
                ['Sunny', 'Warm', '?', 'Strong', '?', '?']
                where '?' means any value is accepted for that attribute
=============================================================================
"""

import pandas as pd


def find_s_algorithm(file_path):
    data = pd.read_csv(file_path)

    print("Training data:")
    print(data)

    attributes = data.columns[:-1]
    class_label = data.columns[-1]

    hypothesis = ['?' for _ in attributes]

    for index, row in data.iterrows():
        if row[class_label] == 'Yes':
            for i, value in enumerate(row[attributes]):
                if hypothesis[i] == '?' or hypothesis[i] == value:
                    hypothesis[i] = value
                else:
                    hypothesis[i] = '?'

    return hypothesis


file_path = 'training_data.csv'
hypothesis = find_s_algorithm(file_path)
print("\nThe final hypothesis is:", hypothesis)

# =============================================================================
# Output:
# Training data:
#      Sky AirTemp Humidity    Wind Water Forecast EnjoySport
# 0  Sunny    Warm   Normal  Strong  Warm     Same        Yes
# 1  Sunny    Warm     High  Strong  Warm     Same        Yes
# 2  Rainy    Cold     High  Strong  Warm   Change         No
# 3  Sunny    Warm     High  Strong  Cool   Change        Yes
#
# The final hypothesis is: ['Sunny', 'Warm', '?', 'Strong', '?', '?']
# Explanation:
#   Sky      = Sunny   (all positive examples have Sunny)
#   AirTemp  = Warm    (all positive examples have Warm)
#   Humidity = ?       (Normal in ex1, High in ex2,ex3 → generalised)
#   Wind     = Strong  (all positive examples have Strong)
#   Water    = ?       (Warm in ex1,ex2, Cool in ex3 → generalised)
#   Forecast = ?       (Same in ex1,ex2, Change in ex3 → generalised)
# =============================================================================

