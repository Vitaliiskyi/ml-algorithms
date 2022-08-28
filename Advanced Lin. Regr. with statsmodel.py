import numpy as np
import statsmodels.api as sm

x = [
    [0, 1],
    [5, 1],
    [15, 2],
    [25, 5],
    [35, 11],
    [45, 15],
    [55, 34],
    [55, 34],
    [55, 34],
    [55, 34],
    [55, 34],
    [55, 34],
    [60, 35] 
    ]

y = [4, 5, 20, 15, 35, 21, 35, 45, 48, 42, 89, 40, 62]
x, y = np.array(x), np.array(y)

# We need to add the column of ones
# to calculate the untercept b0
x = sm.add_constant(x)
print(f"New input array x: {x}")

# Instance of the class statsmodels.regression.linear_model.OLS.
# Create a model
model = sm.OLS(y, x) # Notice that the first argument is the output

#instance of the class statsmodels.regression.linear_model.RegressionResultsWrapper.
results = model.fit()

# Get results
# to get the table with the results of linear regression
print(results.summary())

print('=============================')
print(f"coefficient of determination 𝑅²: {results.rsquared}")
print('=============================')
print(f"adjusted coefficient of determination: {results.rsquared_adj}")
print('=============================')
print(f"regression coefficients(𝑏₀, 𝑏₁, 𝑏₂): {results.params}")

# Predict response
# We can obtain yhe predicted response on
# the input values, using .fittedvalues or .predict()
# with the input array as the argument
print(f"predicted response:\n{results.fittedvalues}")
print(f"predicted response:\n{results.predict(x)}")

# x_new - new input arguments
x_new = ([
    [2, 5],
    [3, 6],
    [6,9],
    [12, 19],
    [23, 56],
    [12,4]
])

x_new = sm.add_constant(x_new)

y_new = results.predict(x_new)
print(y_new)











