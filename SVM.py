# SVM Classification
from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]

clf = svm.SVC()
clf.fit(X, y)

# Some properties of these support vectors, can be found
# in attributes support_vectors_, support_ and n_support_

# Get support vectors
clf.support_vectors_
print(f"Properties of these supp. vect.: {clf.support_vectors_}")

# Get indices of support vectors
clf.support_
print(f"Indices of support vectors: {clf.support_}")

# Get number of support vectors for each class
clf.n_support_
print(f"Numbers of support vectors for each class: {clf.n_support_}")

# After being fitted, the model can be used to predict new val.
new_predict = clf.predict([[2., 2.]])
print(f"New prediction: {new_predict}")


