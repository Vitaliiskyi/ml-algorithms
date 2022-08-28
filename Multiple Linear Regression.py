import numpy as np
from sklearn.linear_model import LinearRegression

x = [[0,1],[5,1],[15,2],[25,2],[35,11],[45,15],[55,34],[60,35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x = np.array(x)
y = np.array(y)

# Create a model and fit it
model = LinearRegression()
model.fit(x, y)

# We cad obtain the properties of the model
r_sq = model.score(x, y)
print(f"coefficient if determination: {r_sq}")

print(f"intercept: {model.intercept_}")

print(f"coefficients: {model.coef_}")

# intercept_ holds the bias b0
# .coef_ its array containing b1 and b2
# intercept is approx. 5.77 when x1=x2=0
# inc. x1 by 1 rise of the predicted responce by 0.43
# similarly x2 by 1 -/-/- by 0.28

# Predict response
y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")

# Equivalent to the following
y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
print(f"predicted response:\n{y_pred}")

# We cad apply this model to new data as well
x_new =np.arange(10).reshape((-1, 2))
y_new = model.predict(x_new)
print(f"predicted response(new predict):\n{y_new}")

