import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('list2/data.csv')

features = data.iloc[:, 0:2].values
labels = data.iloc[:, 2].values

plt.figure(figsize=(10, 6))

for label in np.unique(labels):
    plt.scatter(features[labels == label][:, 0], features[labels == label][:, 1], label=label)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
