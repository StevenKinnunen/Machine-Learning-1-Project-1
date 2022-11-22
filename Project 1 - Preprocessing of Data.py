import pandas as pd
import numpy as np
import sklearn as sk
import math
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

Housing_df = pd.read_csv("HousingDataSet.csv")

#Checking Data Types
Housing_df.info()  #Date, bathrooms and floors have incorrect data types

#Checked for missing values using isnull()
print(Housing_df.isnull().sum())

#Changed 'date' from object to datetime
Housing_df['date']=pd.to_datetime(Housing_df['date'].astype(str), format='%Y/%m/%d')

#Changed format to M-D-Y
Housing_df['date'] = Housing_df['date'].dt.strftime('%m-%d-%Y')
#THIS PART CHANGES THE FORMAT, BUT ALSO CHANGES DATATYPE BACK TO OBJECT? UNSURE WHY
Housing_df.info()

#Checking range of values in both bathroom and floor
Housing_df['bathrooms'].unique()
Housing_df['floors'].unique()

#Changing data type to Integers for the columns bathrooms and floors
Housing_df[['bathrooms', 'floors']] = Housing_df[['bathrooms', 'floors']].astype(int)

#Changes are reflected, will also now drop unnecessary columns:'yr_renovated','sqft_living15' & 'sqft_lot15' columns
Housing_df.head()

#dropping 'yr_renovated','sqft_living15','sqft_lot15'
Housing_df = Housing_df.drop(['yr_renovated','sqft_living15','sqft_lot15'],axis=1)

#reviewing descriptive statistics to check for outliers or other inconsistencies
Housing_df.describe()

#Keeping rows with less than 15 bedrooms, (33 bedrooms seems unrealistic)
Housing_df = Housing_df.loc[Housing_df["bedrooms"] < 15]

#All values in sqft_basement that have basement (>=1) = 1
Housing_df['sqft_basement'].mask(Housing_df['sqft_basement'] >=1 ,1, inplace=True)

#max bedrooms is now 11 and sqft_basement reflects changes
Housing_df.describe()

#lastly, we need to convert 'waterfront','view' and 'sqft_basement' into categorical data types
Housing_df[['waterfront','view','sqft_basement']] = Housing_df[['waterfront','view','sqft_basement']].astype('category')

Housing_df.dtypes #all data types are now correct, EXCEPT date (should be datetime)

#Changed column name of 'sqft_basement' to just 'basement'
Housing_df.rename(columns={"sqft_basement":"basement"})

print(Housing_df) #all changes are reflected

#Normalize data
scaler = preprocessing.MinMaxScaler()
names = Housing_df.columns
d = scaler.fit_transform(Housing_df)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()

# For the regression, we need to predict another price value. To do so, we will assign price to y and all other columns to X.
y = scaled_df['price']
X = scaled_df.drop(['price'], axis = 1)

# Descriptive statistics
X.describe().T

#Split the data into training (70%) and testing (30%)
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

#feature scaling - Double check if we need this step (if we dont run this step the error change)
scaler = StandardScaler()
# Fit only on X_train
scaler.fit(X_train)
# Scale both X_train and X_test
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Recreate a dataframe - Idem before double check if we need this step. 
col_names=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode']
scaled_df_2 = pd.DataFrame(X_train, columns=col_names)
scaled_df_2.describe().T

#Training and Predicting KNN Regression
#K=5
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(X_train, y_train)

#Make predictions
y_pred = regressor.predict(X_test)

#Calculate error
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')

#Training and Predicting KNN Regression
#K=10
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=10)
regressor.fit(X_train, y_train)

#Make predictions
y_pred = regressor.predict(X_test)


#Calculate error
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'mae: {mae}')
print(f'mse: {mse}')
print(f'rmse: {rmse}')

