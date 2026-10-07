from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


iris = load_iris()
X, y = iris.data, iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


print("Predictions:", knn.predict(X_test))
print("Actual:     ", y_test)


result = knn.predict([[2, 4, 6, 2]])
print("New flower:", iris.target_names[result]) 
