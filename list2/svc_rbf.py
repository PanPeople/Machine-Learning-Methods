import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('list2/data.csv')

features = data.iloc[:, 0:2].values
labels = data.iloc[:, 2].values

results = []

for kernel in ['rbf']:
    for C in np.arange(0.1,10,0.1):
        for gamma in ['scale']:
            model = SVC(kernel=kernel, C=C, gamma=gamma)
            model.fit(features, labels)
            predictions = model.predict(features)
            accuracy = accuracy_score(labels, predictions)
            results.append(accuracy)


plt.plot(np.arange(0.1,10,0.1), results)

plt.xlabel('C')
plt.ylabel('Score')
plt.title('RBF')
# plt.legend()
plt.xticks(np.arange(0, 10, 0.5))
plt.grid()
plt.show()
