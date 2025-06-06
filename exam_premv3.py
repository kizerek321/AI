import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier
from matplotlib.colors import ListedColormap

# --- Configuration & Helper Functions ---

# Consistent color mapping for classes
CMAP_POINTS = ListedColormap(['#FF0000', '#0000FF']) # Red for class 0, Blue for class 1
CMAP_REGIONS = ListedColormap(['#FFAAAA', '#AAAAFF']) # Lighter shades for regions

def plot_decision_boundary(ax, clf, X, y, title, classifier_name="Classifier"):
    """Plots the decision boundary for a given classifier."""
    X = np.array(X)
    y = np.array(y)

    # Plot data points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP_POINTS, s=50, edgecolors='k', zorder=2)

    # Create a mesh to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions on the mesh
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(grid_points)
    else:
        Z = clf.predict_proba(grid_points)[:, 1] # Probability of class 1
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and regions
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=CMAP_REGIONS, levels=np.linspace(Z.min(), Z.max(), 3), zorder=1)
    contour_levels = sorted(list(set([-1, 0, 1] if hasattr(clf, "decision_function") and classifier_name == "SVM" else [0.5]))) # 0.5 for probability
    if classifier_name == "SVM":
        contour_lines = ['--', '-', '--']
    else:
        contour_lines = ['-']

    ax.contour(xx, yy, Z, colors='k', levels=contour_levels, alpha=0.7, linestyles=contour_lines, zorder=3)


    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle='--', alpha=0.7)

    # Specific to SVM: plot support vectors
    if classifier_name == "SVM" and hasattr(clf, 'support_vectors_'):
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=100, linewidth=1, facecolors='none', edgecolors='k', zorder=4, label='Support Vectors')
        ax.legend()

    # Specific to Perceptron: highlight misclassified points and print learned function
    if classifier_name == "Perceptron":
        predictions = clf.predict(X)
        misclassified = (predictions != y)
        if np.any(misclassified):
            ax.scatter(X[misclassified, 0], X[misclassified, 1], facecolors='none',
                       edgecolors='lime', linewidths=2, s=100, label='Misclassified', zorder=5)
        ax.legend()
        try:
            w = clf.coef_[0]
            b = clf.intercept_[0]
            if w[1] != 0: # Avoid division by zero
                slope = -w[0] / w[1]
                intercept = -b / w[1]
                print(f"Perceptron Learned function: y = {slope:.2f}x + ({intercept:.2f})")
            else:
                # Vertical line x = -b/w[0] or no simple y=mx+b form
                if w[0] != 0:
                    x_intercept = -b / w[0]
                    print(f"Perceptron Learned function: x = {x_intercept:.2f} (Vertical Line)")
                    ax.axvline(x_intercept, color='k', linestyle='-', label='Decision boundary (Perceptron)')
                else:
                    print("Perceptron Learned function: Cannot represent as y=mx+b (zero weights)")
        except Exception as e:
            print(f"Could not calculate Perceptron decision boundary equation: {e}")


# --- Classifier Definitions ---

def get_classifiers():
    """Returns a dictionary of classifiers to be used."""
    return {
        "SVM": svm.SVC(kernel='linear', probability=True), # probability=True for decision_function consistency if needed
        "Perceptron": Perceptron(max_iter=1000, tol=1e-4, random_state=42), # tol and random_state for reproducibility
        "Decision Tree (Depth 2)": DecisionTreeClassifier(max_depth=2, random_state=42),
        "AdaBoost (20 estimators)": AdaBoostClassifier(n_estimators=20, random_state=1)
    }

# --- Main Execution ---

if __name__ == '__main__':
    # Training data
    X_train = np.array([
        [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],[2.0, 0.0], [3.0, 3.0], [4.0, 3.0],
        [1.0, 2.0], [1.0, 3.0], [2.0, 2.0], [2.0, 3.0], [3.0, 1.0],
        [4.0, 1.0]
    ])
    y_train = np.array([1, 1, 1, 1, 1,1, 0, 0, 0, 0, 0, 0]) # 0 for red, 1 for blue

    classifiers = get_classifiers()
    num_classifiers = len(classifiers)

    # Create a figure with subplots
    # Adjust layout if more classifiers are added
    fig, axes = plt.subplots(nrows=2, ncols= (num_classifiers + 1) // 2, figsize=(15, 10))
    axes = axes.ravel() # Flatten the axes array for easy iteration

    for i, (name, clf) in enumerate(classifiers.items()):
        print(f"\n--- Training and Plotting: {name} ---")
        clf.fit(X_train, y_train)

        ax = axes[i]
        plot_decision_boundary(ax, clf, X_train, y_train, name, classifier_name=name.split(" (")[0]) # Pass classifier type

        # Specific to SVM: print decision function
        if name == "SVM":
            try:
                w = clf.coef_[0]
                b = clf.intercept_[0]
                if w[1] != 0: # Avoid division by zero
                    slope = -w[0] / w[1]
                    intercept = -b / w[1]
                    print(f"SVM Decision function: y = {slope:.2f}x + ({intercept:.2f})")
                else:
                     if w[0] != 0:
                        x_intercept = -b / w[0]
                        print(f"SVM Decision function: x = {x_intercept:.2f} (Vertical Line)")
                     else:
                        print("SVM Decision function: Cannot represent as y=mx+b (zero weights)")
            except Exception as e:
                print(f"Could not calculate SVM decision boundary equation: {e}")


        # Specific to Decision Tree: plot the tree structure in a new figure
        if "Decision Tree" in name:
            plt.figure(figsize=(12, 8)) # New figure for the tree
            plot_tree(clf, filled=True, feature_names=["Feature 1", "Feature 2"],
                      class_names=["Class 0 (Red)", "Class 1 (Blue)"], rounded=True, fontsize=10)
            plt.title(f"{name} - Structure")
            plt.show()


    # Remove any unused subplots
    for j in range(num_classifiers, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=3.0) # Add padding between subplots
    plt.suptitle("Classifier Decision Boundaries", fontsize=16, y=1.02) # Add a main title
    plt.show()

    print("\n--- AdaBoost Classifier (Individual Plot for Clarity) ---")
    # AdaBoost often benefits from a slightly different visualization approach or focus
    # Here, we'll reuse plot_decision_boundary but could customize further if needed.
    ada_clf = classifiers["AdaBoost (20 estimators)"] # Already trained

    fig_ada, ax_ada = plt.subplots(figsize=(8, 6))
    plot_decision_boundary(ax_ada, ada_clf, X_train, y_train, "AdaBoost Decision Boundary", classifier_name="AdaBoost")
    plt.show()