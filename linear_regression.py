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

warnings.filterwarnings('ignore')

df = pd.read_excel(r'C:\Users\DARPAN\Documents\College\6th Semester\BSc Project (DSE6)\Data\list_big5(1).xlsx')
print(df)

# Assuming df is your DataFrame
X = df.iloc[:,1:2].values # features
Y = df.iloc[:,2].values # Target variable

print('X', X)
print('Y', Y)


