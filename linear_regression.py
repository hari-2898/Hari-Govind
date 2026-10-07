import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression().fit(X, y)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction:", model.predict([[6]])[0])
