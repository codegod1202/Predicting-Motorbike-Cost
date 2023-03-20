import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#importing libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#importing the library for using regular expressions
import re

data = pd.read_csv('/kaggle/input/motorbike-cost/train.csv')
test_data = pd.read_csv('../input/motorbike-cost/test.csv')

#Taking an initial look at the data and its basic properties
data.head()
test_data.head()
data.shape
test_data.shape
data.info()
data.describe()        
data.columns

bike_id = data['bike_id']
data = data.drop('bike_id', axis=1)
test_bike_id = test_data['bike_id']
test_data = test_data.drop('bike_id', axis=1)
data.head()

#Fixing the problem of null values in the data
data.isnull().sum()
test_data.isnull().sum()

num_cols = data.select_dtypes(['int64', 'float64'])
obj_cols = data.select_dtypes('object')
print(num_cols.columns)
print(obj_cols.columns)

for col in obj_cols:
    if data[col].isnull().sum() > 0:
        data_mostFrequent = data[col].describe()['top']
        data[col] = data[col].fillna(data_mostFrequent)
        test_data[col] = test_data[col].fillna(data_mostFrequent)
       
#test_data.shape
#data.isnull().sum()
#test_data.isnull().sum()

#Retrieving obvious int and float data from object type
data['kms_driven'].value_counts()
test_data['kms_driven'].value_counts()

data_kms = data['kms_driven'].loc[data['kms_driven'].str[0].str.isdigit()==True]
data_kms = data_kms.str.strip(' Km').astype('int64')
mean_kms_obj = str(int(data_kms.describe()['mean'])) + " Km"
mean_kms_obj

data['kms_driven'].loc[data['kms_driven'].str[0].str.isalpha()==True] = mean_kms_obj
test_data['kms_driven'].loc[test_data['kms_driven'].str[0].str.isalpha()==True] = mean_kms_obj

#data['kms_driven'].value_counts()
#test_data['kms_driven'].value_counts()
data['kms_driven'] = data['kms_driven'].str.strip(' Km').astype('int64')
test_data['kms_driven'] = test_data['kms_driven'].str.strip(' Km').astype('int64')
#data['kms_driven']
#test_data['kms_driven']
#test_data.shape
def getInt(m):
    ints = re.findall(r'\d+', str(m))
    if len(ints) == 0:
        return 0
    return int(ints[0])

def getFloat(m):
    floats = re.findall(r'\d+\.\d+', str(m))
    if len(floats) == 0:
        floats = re.findall(r'\d+', str(m))
        if len(floats) == 0:
            return 0.0
        return float(floats[0])
    return float(floats[0])
data['mileage'] = data['mileage'].apply(lambda m: getInt(m))
#data['mileage']
test_data['mileage'] = test_data['mileage'].apply(lambda m: getInt(m))
#test_data['mileage']
data['power'] = data['power'].apply(lambda m: getFloat(m))
#data['power']
test_data['power'] = test_data['power'].apply(lambda m: getFloat(m))
#test_data['power']

#Taking a look at the object data type
obj_cols = data.select_dtypes('object')
obj_cols.columns

data['model_name'].value_counts()
data['owner'].value_counts()
data['location'].value_counts()
data['price']


#Data Visualization (Finding and Confirming Trends)
sns.lmplot(x='model_year', y='price', data=data)
sns.lmplot(x='kms_driven', y='price', data=data)
sns.lmplot(x='power', y='price', data=data)
sns.catplot(data=data, x='owner', y='price', jitter=False)
plt.xticks(rotation = 45, ha = 'right')

#Scaling Features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data[['model_year', 'kms_driven', 'mileage', 'power']] = scaler.fit_transform(data[['model_year', 'kms_driven', 'mileage', 'power']])
test_data[['model_year', 'kms_driven', 'mileage', 'power']] = scaler.transform(test_data[['model_year', 'kms_driven', 'mileage', 'power']])

#Defining our model
import catboost as cb
from sklearn.model_selection import train_test_split

y = data['price']
data = data.drop(['price'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(data, y, test_size=0.1, random_state=42)
train_dataset = cb.Pool(X_train, y_train, cat_features=['model_name', 'owner', 'location'])
valid_dataset = cb.Pool(X_valid, y_valid, cat_features=['model_name', 'owner', 'location'])

#Training our model
model = cb.CatBoostRegressor(loss_function='RMSE', verbose=False)
model.fit(train_dataset)

#Making predictions on validation set and evaluating performance
from sklearn.metrics import mean_squared_error
preds = model.predict(X_valid)
rmse = (np.sqrt(np.mean(np.square(np.log(y_valid+1)-np.log(preds+1)))))
rmse

#Creating the final submission file
test_preds = model.predict(test_data)
test_bike_id = np.array(test_bike_id)
submission = pd.DataFrame(data=[test_bike_id, test_preds]).T
submission.columns = ['bike_id', 'price']
submission['bike_id'] = submission['bike_id'].astype('int64')
submission.set_index('bike_id', inplace=True)
submission.head()

submission.to_csv('submission.csv')
submission.shape
