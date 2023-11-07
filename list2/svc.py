import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('list2/data.csv')

features = data.iloc[:, 0:2].values
labels = data.iloc[:, 2].values

plt.figure(figsize=(10, 6))

for label in np.unique(labels):
    plt.scatter(features[labels == label][:, 0], features[labels == label][:, 1], label=label)



results = {}

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    for C in [0.1, 0.5, 1, 2, 5, 7, 10, 15, 20]:
        for gamma in ['scale', 'auto']:
            degrees = range(1,9) if kernel == 'poly' else [3]
            for degree in degrees:    # for polynomyal, default 3, ignored in all others
                model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
                model.fit(features, labels)
                predictions = model.predict(features)
                accuracy = accuracy_score(labels, predictions)
                results[(kernel, C, degree, gamma)] = accuracy
                print(f"{kernel}, C={C}, deg={degree}, gamma={gamma}: {accuracy:.3f}")

print(f"\n\n\n")

for (result, accuracy) in sorted(results.items(), key=lambda x: x[1]):
    (kernel, C, degree, gamma) = result
    print(f"{kernel}, C={C}, deg={degree}, gamma={gamma}: {accuracy:.3f}")
best_params = max(results, key=results.get)
best_accuracy = results[best_params]

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()
