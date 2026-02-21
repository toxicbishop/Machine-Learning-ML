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

"""Question: For a given set of training data examples stored in a .CSV file, implement and demonstrate the Find-S 
algorithm to output a description of the set of all hypotheses consistent with the training examples."""

import os

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


file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.csv')
hypothesis = find_s_algorithm(file_path)
print("\nThe final hypothesis is:", hypothesis)

"""
=============================================================================
Output:
Training data:

     Outlook  Temperature  Humidity  Windy  PlayTennis
0     Sunny         Hot     High     False       No
1     Sunny         Hot     High     True        No
2  Overcast         Hot     High     False       Yes
3      Rain        Cold     High     False       Yes
4      Rain        Cold     High     True        No
5  Overcast         Hot     High     True        Yes
6     Sunny         Hot     High     False       No


The final hypothesis is: ['Overcast', 'Hot', 'High', '?']
=============================================================================
"""

