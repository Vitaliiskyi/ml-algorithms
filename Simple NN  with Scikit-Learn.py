from sklearn.neural_network import MLPClassifier
import numpy as np

# input data, sweet or bitter survey, good(1) or bad(0)
survey = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
    ])

features_train = survey[:, 0:2]
labels_train = survey[:, 2]

# Define the model
mlp = MLPClassifier(hidden_layer_sizes=5, max_iter=3000)

mlp.fit(features_train, labels_train)
features_train
labels_train
print(f"Training set score: {mlp.score(features_train, labels_train):.3%}")
print(f"Testing set score: {mlp.score(features_test, labels_test):.3%}\n")
