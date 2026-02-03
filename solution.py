import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset (features + target labels).
iris_dataset = load_iris()
# Use class labels for coloring points on plots.
colors = iris_dataset.target

# 1) (sepal length, petal length, petal width) -> drop sepal width (index 1)
iris_dataset_1 = np.delete(iris_dataset.data, 1, axis=1)

# 2) (sepal width, petal length, petal width) -> drop sepal length (index 0)
iris_dataset_2 = np.delete(iris_dataset.data, 0, axis=1)

# Base dataset from the lesson: only petal length and petal width.
iris_dataset_base = iris_dataset.data[:, 2:4]

# Train/test split for dataset 1.
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
    iris_dataset_1,
    iris_dataset.target,
    random_state=17,
)

# Train/test split for dataset 2.
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(
    iris_dataset_2,
    iris_dataset.target,
    random_state=17,
)

# Train a KNN model for dataset 1.
knn_1 = KNeighborsClassifier(n_neighbors=5)
knn_1.fit(x_train_1, y_train_1)

# Train a KNN model for dataset 2.
knn_2 = KNeighborsClassifier(n_neighbors=5)
knn_2.fit(x_train_2, y_train_2)

# Train/test split for the 2-feature baseline dataset.
x_train_base, x_test_base, y_train_base, y_test_base = train_test_split(
    iris_dataset_base,
    iris_dataset.target,
    random_state=17,
)
# Train a KNN model for the baseline dataset.
knn_base = KNeighborsClassifier(n_neighbors=5)
knn_base.fit(x_train_base, y_train_base)

# Compute accuracy for each model on its test set.
accuracy_1 = accuracy_score(y_test_1, knn_1.predict(x_test_1))
accuracy_2 = accuracy_score(y_test_2, knn_2.predict(x_test_2))
accuracy_base = accuracy_score(y_test_base, knn_base.predict(x_test_base))

# Track the best accuracy and which neighbor counts achieve it.
best_accuracy = -1.0
best_neighbors = []

# Sweep n_neighbors from 1 to 20 for dataset 1 and keep the best result(s).
for n_neighbors in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train_1, y_train_1)
    current_accuracy = accuracy_score(y_test_1, knn.predict(x_test_1))
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_neighbors = [n_neighbors]
    elif current_accuracy == best_accuracy:
        best_neighbors.append(n_neighbors)

# Print accuracies for 3-feature datasets and the baseline.
print(
    f"Accuracy_1: {accuracy_1}, accuracy_2: {accuracy_2}, "
    f"accuracy_base: {accuracy_base}"
)
# Print the best neighbor values for dataset 1.
print(
    f"Best n_neighbors for iris_dataset_1: {best_neighbors} "
    f"with accuracy: {best_accuracy}"
)

# 3D scatter plot for dataset 1.
ax = plt.axes(projection="3d")
ax.scatter3D(
    iris_dataset_1[:, 0],
    iris_dataset_1[:, 1],
    iris_dataset_1[:, 2],
    alpha=0.8,
    c=colors,
)
ax.set_xlabel("sepal length (cm)")
ax.set_ylabel("petal length (cm)")
ax.set_zlabel("petal width (cm)")
ax.set_title("Iris: sepal length + petal length + petal width")

# 3D scatter plot for dataset 2.
ax = plt.axes(projection="3d")
ax.scatter3D(
    iris_dataset_2[:, 0],
    iris_dataset_2[:, 1],
    iris_dataset_2[:, 2],
    alpha=0.8,
    c=colors,
)
ax.set_xlabel("sepal width (cm)")
ax.set_ylabel("petal length (cm)")
ax.set_zlabel("petal width (cm)")
ax.set_title("Iris: sepal width + petal length + petal width")

plt.show()
