import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# Create a random dataset
# rng = np.random.RandomState(1)
# X = np.sort(200 * rng.rand(600, 1) - 100, axis=0)
# y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
# y += 0.5 - rng.rand(*y.shape)

dfX = pd.read_csv('x_features2.csv')
dfY = pd.read_csv('y_features2.csv')


X = dfX.to_numpy()
y = dfY.to_numpy()

y = y[:, 1:]



# print("X =", X)
# print("y =", y)
# print(type(X))
# print(type(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7143, test_size=0.2857, stratify=y
)

# print("X train =", X_train)
# print("y train =", y_train)
# print("X test =", X_test)
# print("y test =", y_test)

max_depth = 30
regr_multirf = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, max_depth=None, random_state=0)
)
regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)

mse = mean_squared_error(y_test, regr_multirf.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

print(y_multirf)
print(y_rf)

print("Y TEST =")
print(y_test)

print("Y MULTIRF =")
print(y_multirf)

print("Y RF =")
print(y_rf)

rmse_rf = mean_squared_error(y_test, regr_rf.predict(X_test), squared=False)
print("The mean squared error (MSE) on test set using RF: {:.4f}".format(rmse_rf))
rmse_mrf = mean_squared_error(y_test, regr_multirf.predict(X_test), squared=False)
print("The mean squared error (MSE) on test set using multi RF: {:.4f}".format(rmse_mrf))

oo, co, eo, ao, no = [],[],[],[],[]
orf, crf, erf, arf, nrf = [],[],[],[],[]
omrf, cmrf, emrf, amrf, nmrf = [],[],[],[],[]

# 5,4,1,2,3
for e in y_test:
    oo.append(e[4])
    co.append(e[3])
    eo.append(e[0])
    ao.append(e[1])
    no.append(e[2])

for e in y_rf:
    orf.append(e[4])
    crf.append(e[3])
    erf.append(e[0])
    arf.append(e[1])
    nrf.append(e[2])

for e in y_multirf:
    omrf.append(e[4])
    cmrf.append(e[3])
    emrf.append(e[0])
    amrf.append(e[1])
    nmrf.append(e[2])

rmse_rf = mean_squared_error(oo, orf, squared=False)
print("The mean squared error (MSE) on test set using RF for Openness: {:.4f}".format(rmse_rf))
rmse_mrf = mean_squared_error(oo, omrf, squared=False)
print("The mean squared error (MSE) on test set using multi RF for Openness: {:.4f}".format(rmse_mrf))

rmse_rf = mean_squared_error(co, crf, squared=False)
print("The mean squared error (MSE) on test set using RF for Conscientiousness: {:.4f}".format(rmse_rf))
rmse_mrf = mean_squared_error(co, cmrf, squared=False)
print("The mean squared error (MSE) on test set using multi RF for Conscientiousness: {:.4f}".format(rmse_mrf))

rmse_rf = mean_squared_error(eo, erf, squared=False)
print("The mean squared error (MSE) on test set using RF for Extraversion: {:.4f}".format(rmse_rf))
rmse_mrf = mean_squared_error(eo, emrf, squared=False)
print("The mean squared error (MSE) on test set using multi RF for Extraversion: {:.4f}".format(rmse_mrf))

rmse_rf = mean_squared_error(ao, arf, squared=False)
print("The mean squared error (MSE) on test set using RF for Agreeableness: {:.4f}".format(rmse_rf))
rmse_mrf = mean_squared_error(ao, amrf, squared=False)
print("The mean squared error (MSE) on test set using multi RF for Agreeableness: {:.4f}".format(rmse_mrf))

rmse_rf = mean_squared_error(no, nrf, squared=False)
print("The mean squared error (MSE) on test set using RF for Neuroticism: {:.4f}".format(rmse_rf))
rmse_mrf = mean_squared_error(no, nmrf, squared=False)
print("The mean squared error (MSE) on test set using multi RF for Neuroticism: {:.4f}".format(rmse_mrf))




N = 15
ind = np.arange(N)  
width = 0.25
  
# xvals = [8, 9, 2] 

xvals = []
for e in y_test:
    xvals.append(e[4])
print("xvals = ", xvals)
xvals = xvals[:N]
bar1 = plt.bar(ind, xvals, width, color = 'r') 
  
# yvals = [10, 20, 30] 
yvals = []
for e in y_rf:
    yvals.append(e[4])
yvals = yvals[:N]

print("yvals =", yvals)
bar2 = plt.bar(ind+width, yvals, width, color='g') 
  
# zvals = [11, 12, 13]
zvals = []
for e in y_multirf:
    zvals.append(e[4])
zvals = zvals[:N]

print("zvals =", zvals)
bar3 = plt.bar(ind+width*2, zvals, width, color = 'b') 

plt.xlabel('Individuals') 
plt.ylabel('Normalized Openness score') 
# plt.title("Players Score")
  
plt.xticks(ind+width,[i for i in range(1,16)]) 
plt.legend( (bar1, bar2, bar3), ('Original Score', 'Predicted score using RandomForestRegressor', 'Predicted score using MultiOutputRegressor') ) 
plt.show() 
