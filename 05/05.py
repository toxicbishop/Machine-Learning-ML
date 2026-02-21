"""
=============================================================================
  MACHINE LEARNING — Week 05
  Topic : Instance-Based Learning — k-Nearest Neighbors (k-NN) Classifier
  File  : 05.py

  Approach : k-NN Classification
    1. Generate 100 random 1-D data points
    2. Label first 50 points: Class1 if x <= 0.5, else Class2
    3. Classify remaining 50 test points using k-NN with Euclidean distance
    4. Evaluate results for multiple values of k = [1, 2, 3, 4, 5, 20, 30]
    5. Visualize classification results with scatter plots

  No external API keys required.
  Dependencies : numpy, matplotlib  (pip install numpy matplotlib)

  Output :
    - Console : classified label for each of the 50 test points per k
                e.g. Point x51 (value: 0.7431) is classified as Class2
    - Plots   : scatter plot per k value showing training data (row 0)
                and classified test points (row 1) in blue / red
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

data = np.random.rand(100)

labels = ["Class1" if x <= 0.5 else "Class2" for x in data[:50]]


def euclidean_distance(x1, x2):
    return abs(x1 - x2)


def knn_classifier(train_data, train_labels, test_point, k):
    distances = [(euclidean_distance(test_point, train_data[i]), train_labels[i]) for i in range(len(train_data))]

    distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = distances[:k]

    k_nearest_labels = [label for _, label in k_nearest_neighbors]

    return Counter(k_nearest_labels).most_common(1)[0][0]


train_data = data[:50]
train_labels = labels

test_data = data[50:]

k_values = [1, 2, 3, 4, 5, 20, 30]

print("--- k-Nearest Neighbors Classification ---")
print("Training dataset: First 50 points labeled based on the rule (x <= 0.5 -> Class1, x > 0.5 -> Class2)")
print("Testing dataset: Remaining 50 points to be classified\n")

results = {}

for k in k_values:
    print(f"Results for k = {k}:")
    classified_labels = [knn_classifier(train_data, train_labels, test_point, k) for test_point in test_data]
    results[k] = classified_labels

    for i, label in enumerate(classified_labels, start=51):
        print(f"Point x{i} (value: {test_data[i - 51]:.4f}) is classified as {label}")
    print("\n")

print("Classification complete.\n")

for k in k_values:
    classified_labels = results[k]
    class1_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i] == "Class1"]
    class2_points = [test_data[i] for i in range(len(test_data)) if classified_labels[i] == "Class2"]

    plt.figure(figsize=(10, 6))
    plt.scatter(train_data, [0] * len(train_data), c=["blue" if label == "Class1" else "red" for label in train_labels],
                label="Training Data", marker="o")
    plt.scatter(class1_points, [1] * len(class1_points), c="blue", label="Class1 (Test)", marker="x")
    plt.scatter(class2_points, [1] * len(class2_points), c="red", label="Class2 (Test)", marker="x")

    plt.title(f"k-NN Classification Results for k = {k}")
    plt.xlabel("Data Points")
    plt.ylabel("Classification Level")
    plt.legend()
    plt.grid(True)
    plt.show()

"""
=============================================================================
Output:
--- k-Nearest Neighbors Classification ---
Training dataset: First 50 points labeled based on the rule (x <= 0.5 -> Class1, x > 0.5 -> Class2)
Testing dataset: Remaining 50 points to be classified

Results for k = 1:
Point x51 (value: 0.2059) is classified as Class1
Point x52 (value: 0.2535) is classified as Class1
Point x53 (value: 0.4856) is classified as Class1
Point x54 (value: 0.9651) is classified as Class2
Point x55 (value: 0.3906) is classified as Class1
Point x56 (value: 0.8903) is classified as Class2
Point x57 (value: 0.9695) is classified as Class2
Point x58 (value: 0.2206) is classified as Class1
Point x59 (value: 0.0203) is classified as Class1
Point x60 (value: 0.1619) is classified as Class1
Point x61 (value: 0.6461) is classified as Class2
Point x62 (value: 0.6523) is classified as Class2
Point x63 (value: 0.8728) is classified as Class2
Point x64 (value: 0.5435) is classified as Class2
Point x65 (value: 0.8246) is classified as Class2
Point x66 (value: 0.9347) is classified as Class2
Point x67 (value: 0.5361) is classified as Class2
Point x68 (value: 0.7215) is classified as Class2
Point x69 (value: 0.9703) is classified as Class2
Point x70 (value: 0.8764) is classified as Class2
Point x71 (value: 0.7543) is classified as Class2
Point x72 (value: 0.1406) is classified as Class1
Point x73 (value: 0.1349) is classified as Class1
Point x74 (value: 0.9705) is classified as Class2
Point x75 (value: 0.2985) is classified as Class1
Point x76 (value: 0.9948) is classified as Class2
Point x77 (value: 0.4551) is classified as Class1
Point x78 (value: 0.2101) is classified as Class1
Point x79 (value: 0.5542) is classified as Class2
Point x80 (value: 0.3202) is classified as Class1
Point x81 (value: 0.6325) is classified as Class2
Point x82 (value: 0.9345) is classified as Class2
Point x83 (value: 0.0156) is classified as Class1
Point x84 (value: 0.8859) is classified as Class2
Point x85 (value: 0.2495) is classified as Class1
Point x86 (value: 0.6380) is classified as Class2
Point x87 (value: 0.7095) is classified as Class2
Point x88 (value: 0.4259) is classified as Class1
Point x89 (value: 0.0052) is classified as Class1
Point x90 (value: 0.6322) is classified as Class2
Point x91 (value: 0.1701) is classified as Class1
Point x92 (value: 0.3693) is classified as Class1
Point x93 (value: 0.4087) is classified as Class1
Point x94 (value: 0.8103) is classified as Class2
Point x95 (value: 0.0773) is classified as Class1
Point x96 (value: 0.8792) is classified as Class2
Point x97 (value: 0.9138) is classified as Class2
Point x98 (value: 0.5567) is classified as Class2
Point x99 (value: 0.8625) is classified as Class2
Point x100 (value: 0.9363) is classified as Class2


Results for k = 2:
Point x51 (value: 0.2059) is classified as Class1
Point x52 (value: 0.2535) is classified as Class1
Point x53 (value: 0.4856) is classified as Class1
Point x54 (value: 0.9651) is classified as Class2
Point x55 (value: 0.3906) is classified as Class1
Point x56 (value: 0.8903) is classified as Class2
Point x57 (value: 0.9695) is classified as Class2
Point x58 (value: 0.2206) is classified as Class1
Point x59 (value: 0.0203) is classified as Class1
Point x60 (value: 0.1619) is classified as Class1
Point x61 (value: 0.6461) is classified as Class2
Point x62 (value: 0.6523) is classified as Class2
Point x63 (value: 0.8728) is classified as Class2
Point x64 (value: 0.5435) is classified as Class2
Point x65 (value: 0.8246) is classified as Class2
Point x66 (value: 0.9347) is classified as Class2
Point x67 (value: 0.5361) is classified as Class2
Point x68 (value: 0.7215) is classified as Class2
Point x69 (value: 0.9703) is classified as Class2
Point x70 (value: 0.8764) is classified as Class2
Point x71 (value: 0.7543) is classified as Class2
Point x72 (value: 0.1406) is classified as Class1
Point x73 (value: 0.1349) is classified as Class1
Point x74 (value: 0.9705) is classified as Class2
Point x75 (value: 0.2985) is classified as Class1
Point x76 (value: 0.9948) is classified as Class2
Point x77 (value: 0.4551) is classified as Class1
Point x78 (value: 0.2101) is classified as Class1
Point x79 (value: 0.5542) is classified as Class2
Point x80 (value: 0.3202) is classified as Class1
Point x81 (value: 0.6325) is classified as Class2
Point x82 (value: 0.9345) is classified as Class2
Point x83 (value: 0.0156) is classified as Class1
Point x84 (value: 0.8859) is classified as Class2
Point x85 (value: 0.2495) is classified as Class1
Point x86 (value: 0.6380) is classified as Class2
Point x87 (value: 0.7095) is classified as Class2
Point x88 (value: 0.4259) is classified as Class1
Point x89 (value: 0.0052) is classified as Class1
Point x90 (value: 0.6322) is classified as Class2
Point x91 (value: 0.1701) is classified as Class1
Point x92 (value: 0.3693) is classified as Class1
Point x93 (value: 0.4087) is classified as Class1
Point x94 (value: 0.8103) is classified as Class2
Point x95 (value: 0.0773) is classified as Class1
Point x96 (value: 0.8792) is classified as Class2
Point x97 (value: 0.9138) is classified as Class2
Point x98 (value: 0.5567) is classified as Class2
Point x99 (value: 0.8625) is classified as Class2
Point x100 (value: 0.9363) is classified as Class2


Results for k = 3:
Point x51 (value: 0.2059) is classified as Class1
Point x52 (value: 0.2535) is classified as Class1
Point x53 (value: 0.4856) is classified as Class1
Point x54 (value: 0.9651) is classified as Class2
Point x55 (value: 0.3906) is classified as Class1
Point x56 (value: 0.8903) is classified as Class2
Point x57 (value: 0.9695) is classified as Class2
Point x58 (value: 0.2206) is classified as Class1
Point x59 (value: 0.0203) is classified as Class1
Point x60 (value: 0.1619) is classified as Class1
Point x61 (value: 0.6461) is classified as Class2
Point x62 (value: 0.6523) is classified as Class2
Point x63 (value: 0.8728) is classified as Class2
Point x64 (value: 0.5435) is classified as Class2
Point x65 (value: 0.8246) is classified as Class2
Point x66 (value: 0.9347) is classified as Class2
Point x67 (value: 0.5361) is classified as Class2
Point x68 (value: 0.7215) is classified as Class2
Point x69 (value: 0.9703) is classified as Class2
Point x70 (value: 0.8764) is classified as Class2
Point x71 (value: 0.7543) is classified as Class2
Point x72 (value: 0.1406) is classified as Class1
Point x73 (value: 0.1349) is classified as Class1
Point x74 (value: 0.9705) is classified as Class2
Point x75 (value: 0.2985) is classified as Class1
Point x76 (value: 0.9948) is classified as Class2
Point x77 (value: 0.4551) is classified as Class1
Point x78 (value: 0.2101) is classified as Class1
Point x79 (value: 0.5542) is classified as Class2
Point x80 (value: 0.3202) is classified as Class1
Point x81 (value: 0.6325) is classified as Class2
Point x82 (value: 0.9345) is classified as Class2
Point x83 (value: 0.0156) is classified as Class1
Point x84 (value: 0.8859) is classified as Class2
Point x85 (value: 0.2495) is classified as Class1
Point x86 (value: 0.6380) is classified as Class2
Point x87 (value: 0.7095) is classified as Class2
Point x88 (value: 0.4259) is classified as Class1
Point x89 (value: 0.0052) is classified as Class1
Point x90 (value: 0.6322) is classified as Class2
Point x91 (value: 0.1701) is classified as Class1
Point x92 (value: 0.3693) is classified as Class1
Point x93 (value: 0.4087) is classified as Class1
Point x94 (value: 0.8103) is classified as Class2
Point x95 (value: 0.0773) is classified as Class1
Point x96 (value: 0.8792) is classified as Class2
Point x97 (value: 0.9138) is classified as Class2
Point x98 (value: 0.5567) is classified as Class2
Point x99 (value: 0.8625) is classified as Class2
Point x100 (value: 0.9363) is classified as Class2


Results for k = 4:
Point x51 (value: 0.2059) is classified as Class1
Point x52 (value: 0.2535) is classified as Class1
Point x53 (value: 0.4856) is classified as Class1
Point x54 (value: 0.9651) is classified as Class2
Point x55 (value: 0.3906) is classified as Class1
Point x56 (value: 0.8903) is classified as Class2
Point x57 (value: 0.9695) is classified as Class2
Point x58 (value: 0.2206) is classified as Class1
Point x59 (value: 0.0203) is classified as Class1
Point x60 (value: 0.1619) is classified as Class1
Point x61 (value: 0.6461) is classified as Class2
Point x62 (value: 0.6523) is classified as Class2
Point x63 (value: 0.8728) is classified as Class2
Point x64 (value: 0.5435) is classified as Class2
Point x65 (value: 0.8246) is classified as Class2
Point x66 (value: 0.9347) is classified as Class2
Point x67 (value: 0.5361) is classified as Class2
Point x68 (value: 0.7215) is classified as Class2
Point x69 (value: 0.9703) is classified as Class2
Point x70 (value: 0.8764) is classified as Class2
Point x71 (value: 0.7543) is classified as Class2
Point x72 (value: 0.1406) is classified as Class1
Point x73 (value: 0.1349) is classified as Class1
Point x74 (value: 0.9705) is classified as Class2
Point x75 (value: 0.2985) is classified as Class1
Point x76 (value: 0.9948) is classified as Class2
Point x77 (value: 0.4551) is classified as Class1
Point x78 (value: 0.2101) is classified as Class1
Point x79 (value: 0.5542) is classified as Class2
Point x80 (value: 0.3202) is classified as Class1
Point x81 (value: 0.6325) is classified as Class2
Point x82 (value: 0.9345) is classified as Class2
Point x83 (value: 0.0156) is classified as Class1
Point x84 (value: 0.8859) is classified as Class2
Point x85 (value: 0.2495) is classified as Class1
Point x86 (value: 0.6380) is classified as Class2
Point x87 (value: 0.7095) is classified as Class2
Point x88 (value: 0.4259) is classified as Class1
Point x89 (value: 0.0052) is classified as Class1
Point x90 (value: 0.6322) is classified as Class2
Point x91 (value: 0.1701) is classified as Class1
Point x92 (value: 0.3693) is classified as Class1
Point x93 (value: 0.4087) is classified as Class1
Point x94 (value: 0.8103) is classified as Class2
Point x95 (value: 0.0773) is classified as Class1
Point x96 (value: 0.8792) is classified as Class2
Point x97 (value: 0.9138) is classified as Class2
Point x98 (value: 0.5567) is classified as Class2
Point x99 (value: 0.8625) is classified as Class2
Point x100 (value: 0.9363) is classified as Class2


Results for k = 5:
Point x51 (value: 0.2059) is classified as Class1
Point x52 (value: 0.2535) is classified as Class1
Point x53 (value: 0.4856) is classified as Class1
Point x54 (value: 0.9651) is classified as Class2
Point x55 (value: 0.3906) is classified as Class1
Point x56 (value: 0.8903) is classified as Class2
Point x57 (value: 0.9695) is classified as Class2
Point x58 (value: 0.2206) is classified as Class1
Point x59 (value: 0.0203) is classified as Class1
Point x60 (value: 0.1619) is classified as Class1
Point x61 (value: 0.6461) is classified as Class2
Point x62 (value: 0.6523) is classified as Class2
Point x63 (value: 0.8728) is classified as Class2
Point x64 (value: 0.5435) is classified as Class2
Point x65 (value: 0.8246) is classified as Class2
Point x66 (value: 0.9347) is classified as Class2
Point x67 (value: 0.5361) is classified as Class1
Point x68 (value: 0.7215) is classified as Class2
Point x69 (value: 0.9703) is classified as Class2
Point x70 (value: 0.8764) is classified as Class2
Point x71 (value: 0.7543) is classified as Class2
Point x72 (value: 0.1406) is classified as Class1
Point x73 (value: 0.1349) is classified as Class1
Point x74 (value: 0.9705) is classified as Class2
Point x75 (value: 0.2985) is classified as Class1
Point x76 (value: 0.9948) is classified as Class2
Point x77 (value: 0.4551) is classified as Class1
Point x78 (value: 0.2101) is classified as Class1
Point x79 (value: 0.5542) is classified as Class2
Point x80 (value: 0.3202) is classified as Class1
Point x81 (value: 0.6325) is classified as Class2
Point x82 (value: 0.9345) is classified as Class2
Point x83 (value: 0.0156) is classified as Class1
Point x84 (value: 0.8859) is classified as Class2
Point x85 (value: 0.2495) is classified as Class1
Point x86 (value: 0.6380) is classified as Class2
Point x87 (value: 0.7095) is classified as Class2
Point x88 (value: 0.4259) is classified as Class1
Point x89 (value: 0.0052) is classified as Class1
Point x90 (value: 0.6322) is classified as Class2
Point x91 (value: 0.1701) is classified as Class1
Point x92 (value: 0.3693) is classified as Class1
Point x93 (value: 0.4087) is classified as Class1
Point x94 (value: 0.8103) is classified as Class2
Point x95 (value: 0.0773) is classified as Class1
Point x96 (value: 0.8792) is classified as Class2
Point x97 (value: 0.9138) is classified as Class2
Point x98 (value: 0.5567) is classified as Class2
Point x99 (value: 0.8625) is classified as Class2
Point x100 (value: 0.9363) is classified as Class2


Results for k = 20:
Point x51 (value: 0.2059) is classified as Class1
Point x52 (value: 0.2535) is classified as Class1
Point x53 (value: 0.4856) is classified as Class1
Point x54 (value: 0.9651) is classified as Class2
Point x55 (value: 0.3906) is classified as Class1
Point x56 (value: 0.8903) is classified as Class2
Point x57 (value: 0.9695) is classified as Class2
Point x58 (value: 0.2206) is classified as Class1
Point x59 (value: 0.0203) is classified as Class1
Point x60 (value: 0.1619) is classified as Class1
Point x61 (value: 0.6461) is classified as Class2
Point x62 (value: 0.6523) is classified as Class2
Point x63 (value: 0.8728) is classified as Class2
Point x64 (value: 0.5435) is classified as Class1
Point x65 (value: 0.8246) is classified as Class2
Point x66 (value: 0.9347) is classified as Class2
Point x67 (value: 0.5361) is classified as Class1
Point x68 (value: 0.7215) is classified as Class2
Point x69 (value: 0.9703) is classified as Class2
Point x70 (value: 0.8764) is classified as Class2
Point x71 (value: 0.7543) is classified as Class2
Point x72 (value: 0.1406) is classified as Class1
Point x73 (value: 0.1349) is classified as Class1
Point x74 (value: 0.9705) is classified as Class2
Point x75 (value: 0.2985) is classified as Class1
Point x76 (value: 0.9948) is classified as Class2
Point x77 (value: 0.4551) is classified as Class1
Point x78 (value: 0.2101) is classified as Class1
Point x79 (value: 0.5542) is classified as Class1
Point x80 (value: 0.3202) is classified as Class1
Point x81 (value: 0.6325) is classified as Class2
Point x82 (value: 0.9345) is classified as Class2
Point x83 (value: 0.0156) is classified as Class1
Point x84 (value: 0.8859) is classified as Class2
Point x85 (value: 0.2495) is classified as Class1
Point x86 (value: 0.6380) is classified as Class2
Point x87 (value: 0.7095) is classified as Class2
Point x88 (value: 0.4259) is classified as Class1
Point x89 (value: 0.0052) is classified as Class1
Point x90 (value: 0.6322) is classified as Class2
Point x91 (value: 0.1701) is classified as Class1
Point x92 (value: 0.3693) is classified as Class1
Point x93 (value: 0.4087) is classified as Class1
Point x94 (value: 0.8103) is classified as Class2
Point x95 (value: 0.0773) is classified as Class1
Point x96 (value: 0.8792) is classified as Class2
Point x97 (value: 0.9138) is classified as Class2
Point x98 (value: 0.5567) is classified as Class2
Point x99 (value: 0.8625) is classified as Class2
Point x100 (value: 0.9363) is classified as Class2


Results for k = 30:
Point x51 (value: 0.2059) is classified as Class1
Point x52 (value: 0.2535) is classified as Class1
Point x53 (value: 0.4856) is classified as Class1
Point x54 (value: 0.9651) is classified as Class2
Point x55 (value: 0.3906) is classified as Class1
Point x56 (value: 0.8903) is classified as Class2
Point x57 (value: 0.9695) is classified as Class2
Point x58 (value: 0.2206) is classified as Class1
Point x59 (value: 0.0203) is classified as Class1
Point x60 (value: 0.1619) is classified as Class1
Point x61 (value: 0.6461) is classified as Class2
Point x62 (value: 0.6523) is classified as Class2
Point x63 (value: 0.8728) is classified as Class2
Point x64 (value: 0.5435) is classified as Class1
Point x65 (value: 0.8246) is classified as Class2
Point x66 (value: 0.9347) is classified as Class2
Point x67 (value: 0.5361) is classified as Class1
Point x68 (value: 0.7215) is classified as Class2
Point x69 (value: 0.9703) is classified as Class2
Point x70 (value: 0.8764) is classified as Class2
Point x71 (value: 0.7543) is classified as Class2
Point x72 (value: 0.1406) is classified as Class1
Point x73 (value: 0.1349) is classified as Class1
Point x74 (value: 0.9705) is classified as Class2
Point x75 (value: 0.2985) is classified as Class1
Point x76 (value: 0.9948) is classified as Class2
Point x77 (value: 0.4551) is classified as Class1
Point x78 (value: 0.2101) is classified as Class1
Point x79 (value: 0.5542) is classified as Class1
Point x80 (value: 0.3202) is classified as Class1
Point x81 (value: 0.6325) is classified as Class2
Point x82 (value: 0.9345) is classified as Class2
Point x83 (value: 0.0156) is classified as Class1
Point x84 (value: 0.8859) is classified as Class2
Point x85 (value: 0.2495) is classified as Class1
Point x86 (value: 0.6380) is classified as Class2
Point x87 (value: 0.7095) is classified as Class2
Point x88 (value: 0.4259) is classified as Class1
Point x89 (value: 0.0052) is classified as Class1
Point x90 (value: 0.6322) is classified as Class2
Point x91 (value: 0.1701) is classified as Class1
Point x92 (value: 0.3693) is classified as Class1
Point x93 (value: 0.4087) is classified as Class1
Point x94 (value: 0.8103) is classified as Class2
Point x95 (value: 0.0773) is classified as Class1
Point x96 (value: 0.8792) is classified as Class2
Point x97 (value: 0.9138) is classified as Class2
Point x98 (value: 0.5567) is classified as Class1
Point x99 (value: 0.8625) is classified as Class2
Point x100 (value: 0.9363) is classified as Class2


Classification complete.
       Saved in Outputs/
=============================================================================
"""
