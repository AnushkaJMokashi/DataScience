# Data Wrangling 
## Import pandas library
import pandas as pd
## Read CSV
data = pd.read_csv("dirtydata - dirtydata.csv")
data
## Return 1st 5 elements
data.head()
data.shape
## Data statistics
data.describe()
## check no. of null values in each column
data.isnull().sum() 
data.count()
#data.count(axis = 'rows')
data.dtypes
## Replace values with it's absolute
data['Calories'] = data['Calories'].abs()
## Mean of the column
x = data.Calories.mean()
x
## Absolute
data['Calories'] = data['Calories'].abs()
## Replace null values with mean
data['Calories'].fillna(x,inplace = True)
data.isnull().sum()
data.dropna(subset = ["Date"],inplace=True)
data.isnull().sum()
import re
data['Date'] = data['Date'].apply(lambda x: x[:-1] if x.endswith("'") else x)
data['Date'][:5]

data.Date
data['Date'] = data['Date'].apply(lambda x: x.split('/'))
data['Date']
data['Date'] = data['Date'].apply(lambda x: ''.join(x))
data.Date
data['Date'] = pd.to_datetime(data['Date'], format="%Y%m%d")
data['Date']
data['Calories'] = data['Calories'].astype(int)
data['Date'].dtype
data['Date'] = pd.to_datetime(data['Date'])
data.Date.dtype
data.loc[7,'Duration'] 
## TO find minimun
data['Duration'].min()
data.Duration
## TO find maximum
data['Duration'].max()

df['Duration'] = df['Duration'].apply(lambda x: 60 if x>100 else x )
data.Duration
# data.loc[(data['Duration'] > 60 and data['Duration'] < 45).item(), 'Duration'] = 45
data.loc[7,'Duration'] = 45
data.duplicated()
## drop duplicated rows
data.drop_duplicates(inplace = True)
data.to_csv('Dirty_data_preprop')
## NBA Data
data_nba = pd.read_csv("nba.csv")
data_nba
data_nba.shape
data_nba['Position'].value_counts()
data_nba
## Converting to numerics
data_nba['Position'].replace(['SG','PF','PG','SF','C'],[0,1,2,3,4], inplace=True)
data_nba
data_nba.to_csv('Replace_function_preprocess')
## Label Encoding
df2= pd.read_csv("nba.csv")
from sklearn import preprocessing
df2['Position'].unique()
label_encoder = preprocessing.LabelEncoder()
df2['Position'] = label_encoder.fit_transform(df2['Position'])
df2['Position'].unique()
df2.Age.min()
df2.Age.max()
### Quantitative to Categorical
category = pd.cut(df2.Age, bins=[19,25,30,35,40],labels=['A','B','C','D'])
## insert in df2
df2.insert(3, 'Age_Group', category)
df2

df2.to_csv('Preprocessed_nba_csv')

