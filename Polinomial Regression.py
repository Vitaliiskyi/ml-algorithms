# We need to transform the array of
# inputs to include nonlinear terms such as x^2

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Provide data
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

# For polinomial regression
# We need to include x^2 and perhaps other terms 
# as additional features when implementing polinomial regression
transformer = PolynomialFeatures(degree=2, include_bias=True)
transformer.fit(x)
x_ = transformer.transform(x)

# OR x_ = PolinomialFeatures(degree=2, include_bias=False).fit_transform(x)
# array([[   5.,   25.],
#       [  15.,  225.],
#       [  25.,  625.],
#       [  35., 1225.],
#       [  45., 2025.],
#       [  55., 3025.]])

# Create a model and fit it
model = LinearRegression(fit_intercept=False).fit(x_, y)

# We can obtain prop. of the model as in the case of lin. regr
r_sq = model.score(x_, y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"coefficient: {model.coef_}")
print(x_)

# Predict response
y_pred = model.predict(x_)
print(f"predicted response: \n{y_pred}")


