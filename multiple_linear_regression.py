from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

data = load_diabetes(as_frame=True)
df = data.frame

X = df[['bmi', 'bp', 's1', 's5']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
print("R2:", round(r2_score(y_test, y_pred), 2))

new = pd.DataFrame([[0.05, 0.03, 0.02, 0.04]],
                   columns=['bmi', 'bp', 's1', 's5'])

print("Prediction:", round(model.predict(new)[0], 2)) 