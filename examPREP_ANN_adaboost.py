import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



if __name__ == '__main__':
    # Training data
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [2.0, 3.0],
        [4.0, 1.0],
        [4.0, 2.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [2.0, 2.0],
        [3.0, 0.0],
        [3.0, 1.0],
        [3.0, 2.0],

    ]
    labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,0]
    X = np.array(X)
    y = np.array(labels)

    clf_a = MLPClassifier(hidden_layer_sizes=(150, 50), max_iter=5000, random_state=1)
    clf_b = MLPClassifier(hidden_layer_sizes=(), max_iter=5000, random_state=1)
    clf_c = AdaBoostClassifier(
        n_estimators=20,
        random_state=1
    )

    clfs = [clf_a, clf_b, clf_c]
    titles = ['a) 2-layer NN (150 neurons)', 'b) Single-layer NN', 'c) AdaBoost']

    # Train classifiers
    for clf in clfs:
        clf.fit(X, y)

    # Plot decision boundaries
    xx, yy = np.meshgrid(np.linspace(0.5, 4.5, 300), np.linspace(0.5, 4.5, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    plt.figure(figsize=(15, 5))

    for i, clf in enumerate(clfs):
        Z = clf.predict(grid).reshape(xx.shape)

        plt.subplot(1, 4, i + 1)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors='k', cmap=plt.cm.coolwarm)
        plt.title(titles[i])
        plt.xlim(0.5, 4.5)
        plt.ylim(0.5, 3.5)
        plt.grid(True)
        plt.xticks([1, 2, 3,4])
        plt.yticks([1, 2, 3,4])

    plt.tight_layout()
    plt.show()