import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

def svm_on_points(X, labels):
    # Train SVM
    clf = svm.SVC(kernel='linear')
    clf.fit(X, labels)

    # Get the weight vector (w) and bias (b)
    w = clf.coef_[0]
    b = clf.intercept_[0]

    # Calculate slope and intercept of the decision boundary line
    slope = -w[0] / w[1]
    intercept = -b / w[1]

    # Print the function
    print(f"The decision function is: y = {slope:.2f}x + ({intercept:.2f})")

    # Convert to numpy array for plotting
    X = np.array(X)

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plot points
    colors = ['red' if label == 0 else 'blue' for label in labels]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, edgecolors='k')

    # Plot decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    contour = ax.contour(XX, YY, Z, colors='k',
                levels=[-1, 0, 1], alpha=0.7,
                linestyles=['--', '-', '--'])

    # Plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')

    plt.title("SVM Decision Boundary with Support Vectors")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()


def perceptron_on_points(X, labels):
    X = np.array(X)
    clf = Perceptron(max_iter=1000, tol=0.0001)
    clf.fit(X, labels)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    print(f"Learned function: y = {slope:.2f}x + ({intercept:.2f})")

    # Plot points
    colors = ['blue' if label == 1 else 'red' for label in labels]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, edgecolors='k')

    # Plot decision boundary line
    x_vals = np.linspace(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5, 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, 'k-', label='Decision boundary')

    # Highlight misclassified points (optional)
    predictions = clf.predict(X)
    misclassified = (predictions != labels)
    plt.scatter(X[misclassified, 0], X[misclassified, 1], facecolors='none',
                edgecolors='black', linewidths=2, s=100, label='Misclassified')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Perceptron Classification")
    plt.legend()
    plt.grid(True)
    plt.show()


def decision_tree_on_points(X, labels, max_depth=None):
    X = np.array(X)
    labels = np.array(labels)

    # Train the decision tree
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X, labels)

    # Print the tree structure
    from sklearn.tree import plot_tree
    plt.figure(figsize=(10, 6))
    plot_tree(clf, filled=True, feature_names=["x", "y"], class_names=["0", "1"])
    plt.title("Decision Tree Structure")
    plt.show()

    # Plot decision regions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)

    colors = ['red' if label == 0 else 'blue' for label in labels]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k', s=50)

    plt.title("Decision Tree Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Training data
    X = [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [2.0, 2.0],
        [2.0, 3.0],
        [2.0, 0.0],
        [3.0, 1.0],
        [3.0, 2.0],
        [4.0, 1.0],
        [4.0, 0.0],
        [4.0, 2.0]
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    svm_on_points(X, labels)
    perceptron_on_points(X, labels)
    clf = decision_tree_on_points(X, labels, max_depth=2)
