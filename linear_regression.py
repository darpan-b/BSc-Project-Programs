import find_min_bounding_circles_and_rectangles

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error, r2_score


warnings.filterwarnings('ignore')

df_x = pd.read_csv('x_features2.csv')
df_y = pd.read_csv('y_features2.csv')

'''
Trying to predict extraversion based on circularity ratio
'''

xd = df_x.iloc[:,5] # features
yd = df_y.iloc[:,1] # target variable

# Check and handle categorical variables
label_encoder = LabelEncoder()
x_categorical = df_x.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical = df_y.select_dtypes(exclude=['object']).values
xd2 = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values
 
# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
 
# Fit the regressor with x and y data
regressor.fit(xd2, yd)

# Access the OOB Score
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')
 
# Making predictions on the same data or new data
predictions = regressor.predict(xd2)
 
# Evaluating the model
mse = mean_squared_error(yd, predictions)
print(f'Mean Squared Error: {mse}')
 
r2 = r2_score(yd, predictions)
print(f'R-squared: {r2}')


