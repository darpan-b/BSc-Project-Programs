import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

# Create a random dataset
# rng = np.random.RandomState(1)
# X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
# y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
# y += 0.5 - rng.rand(*y.shape)

dfX = pd.read_csv('x_features2.csv')
dfY = pd.read_csv('y_features2.csv')


X = dfX.to_numpy()
y = dfY.to_numpy()

# print("X =", X)
# print("y =", y)
# print(type(X))
# print(type(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7143, test_size=0.2857, stratify=y
)

print("X train =", X_train)
print("y train =", y_train)
print("X test =", X_test)
print("y test =", y_test)

max_depth = 30
regr_multirf = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=0)
)
regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)

mse = mean_squared_error(y_test, regr_multirf.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

print(y_multirf)
print(y_rf)


# Plot the results
plt.figure()
s = 50
a = 0.4
plt.scatter(
    y_test[:, 0],
    y_test[:, 1],
    edgecolor="k",
    c="navy",
    s=s,
    marker="s",
    alpha=a,
    label="Data",
)
plt.scatter(
    y_multirf[:, 0],
    y_multirf[:, 1],
    edgecolor="k",
    c="cornflowerblue",
    s=s,
    alpha=a,
    label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test),
)
plt.scatter(
    y_rf[:, 0],
    y_rf[:, 1],
    edgecolor="k",
    c="c",
    s=s,
    marker="^",
    alpha=a,
    label="RF score=%.2f" % regr_rf.score(X_test, y_test),
)
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and the multi-output meta estimator")
plt.legend()
plt.show()

