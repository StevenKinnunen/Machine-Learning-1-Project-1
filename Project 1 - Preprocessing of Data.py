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
