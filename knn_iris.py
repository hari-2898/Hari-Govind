from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create & train model (k=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Predict on test set
print("Predictions:", knn.predict(X_test))
print("Actual:     ", y_test)

# Predict a new sample
result = knn.predict([[2, 4, 6, 2]])
print("New flower:", iris.target_names[result]) 