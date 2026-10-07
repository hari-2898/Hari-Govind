from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Load data
X, y = load_iris(return_X_y=True)

# Split data (50% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Create & train model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict on test set
y_pred = gnb.predict(X_test)
print("Predictions:", y_pred)

# Predict new sample
y_new = gnb.predict([[5, 5, 4, 4]])
print("Predicted for [[5,5,4,4]]:", y_new)

# Accuracy score
print("Naive Bayes score:", gnb.score(X_test, y_test))