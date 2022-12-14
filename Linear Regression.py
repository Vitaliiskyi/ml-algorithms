import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

#Create an instance of the class LinearRegression
model = LinearRegression()  

# .fit() calculate the optim. val. b0 and b1 
model.fit(x, y)

r_sq = model.score(x, y)
print (f"coefficient of determination: {r_sq}")
print (f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

# The value of b0 (intercept) approx. = 5.63
# Model predicts 5.63 when x = 0
# The value b1 = 0.54 means that the pred. rises by 0.54
# when x incr. by one.

y_pred = model.predict(x)
print(f"predicted response:\n{y_pred}")

#or
y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response;\n{y_pred}")

# New inputs x (New prediction)
x_new = np.arange(15).reshape((-1, 1))
y_new = model.predict(x_new)
print(f"new predicted response;\n{y_new}")





