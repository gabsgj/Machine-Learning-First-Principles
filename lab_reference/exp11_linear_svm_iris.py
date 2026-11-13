"""
Experiment 11: Linear Support Vector Machine and Margin Visualization
=====================================================================
Dataset: Iris (Petal Length & Petal Width, Setosa vs. Non-Setosa)
Identifies support vectors and renders decision boundary with margins.
"""

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Dataset Ingestion (2 Features for Visual Display)
iris = load_iris()
X = iris.data[:, 2:4]
y = (iris.target == 0).astype(int)

# 2. Linear SVM Estimation
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

print("=== Experiment 11: Support Vector Machine ===")
print("Training Accuracy:        ", model.score(X, y))
print("Number of Support Vectors:", len(model.support_vectors_))

# 3. Decision Boundary Rendering
plt.figure(figsize=(7, 4.8), dpi=150)
DecisionBoundaryDisplay.from_estimator(
    model, X, response_method='predict',
    cmap='bwr', alpha=0.3
)

# 4. Data Points & Circled Support Vectors
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', label='Samples')
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=150, facecolors='none', edgecolors='#2ca02c', lw=2,
            label='Support Vectors')

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Experiment 11: Linear SVM Hyperplane & Support Vectors')
plt.legend()
plt.tight_layout()
plt.savefig('exp11_linear_svm.png')
print("Saved visualization to exp11_linear_svm.png")
