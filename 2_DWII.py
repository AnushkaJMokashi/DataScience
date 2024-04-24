# DSBDAL2_DataWranglingII
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

df = pd.read_csv("A1.csv")
df
df.head()
df.shape
df.info()
## Handling Missing Data
### isnull()
df.isnull()
## to find total null values is each column
df.isnull().sum()
### notnull()
df.notnull()
df.notnull().sum()
## Handling the missing values
### fillna()
df.fillna(0,inplace=True)
df
df = pd.read_csv("A1.csv")
df
df.fillna(50,inplace=True)
df
df = pd.read_csv("A1.csv")
df
## Check previous not null value and fills in current null
df.fillna(method='pad')
df = pd.read_csv("A1.csv")
df
## CHecks next notnull value and fills at current
df.fillna(method='bfill')
### interpolate()
df = pd.read_csv("A1.csv")
df
df.interpolate(method='linear',limit_direction = 'forward',inplace=True)
df
df.isnull().sum()
df['Subject 3'].fillna(df['Subject 3'].mean(),inplace=True)
df['Subject 4'].fillna(df['Subject 4'].mean(),inplace=True)
df
df.isnull().sum()
### replace()
df.replace(to_replace=np.nan,value=df['Subject 3'].mean(),inplace=True)
df
## Drop Missing Values
df = pd.read_csv("A1.csv")
df
df.dropna()
df.shape
## in a row if all features have null value then drop
df.dropna(how='all',inplace=True)
df
## any value null drop
df.dropna(how='any',inplace=True)
df
## drop column
df.dropna(axis=1)
## Check for negative Values
df[df[['Subject 1','Subject 2','Subject 3','Subject 4','Attendance']]<0]
df[['Subject 1','Subject 2','Subject 3','Subject 4','Attendance']] = df[['Subject 1','Subject 2','Subject 3','Subject 4','Attendance']].clip(lower=0)
df
## Handling Inconsistencies - Duplicate Data
df.drop_duplicates(inplace=True)
df
## Handling Outliers
### Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
nc = ['Subject 1','Subject 2','Subject 3','Subject 4','Attendance']
nc
for col in nc:
    sns.boxplot(x=df[col])
    plt.title(f'boxplot of {col}')
    plt.show()
   
for column in nc:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=df.index, y=column, label=column)
    
    plt.title(f'Scatter Plot - {column}')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

### Handling Outliers
for col in nc:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lb = Q1 - 1.5*IQR
    ub = Q3 + 1.5*IQR
    df[col] = np.where((df[col]<lb) | (df[col]>ub),df[col].median(),df[col])

df
dfz = pd.read_csv('A1.csv')
dfz
dfz.dropna()
# #z-score manually
# outliers = []
# def detect_outliers_zscore(data):
#     thres = 3
#     median_value = data.median()
#     mean = np.mean(data)
#     std = np.std(data)
#     # print(mean, std)
#     for i in data:
#         z_score = (i-mean)/std
#         if (np.abs(z_score) > thres):
#             i = median_value
#     return data

# for feat in nc:
#     detect_outliers_zscore(dfz[feat])
# from scipy.stats import zscore
# # zscores_df = dfz['Subject 1'].apply(zscore)
# for col in nc:
#     z_scores = df[col].apply(zscore)
#     print(f"Skewness of {col}: {z_scores}")
for col in nc:
    col_zscore = col + '_zscore'
    df[col_zscore] = (df[col] - df[col].mean())/df[col].std(ddof=0)
df

##OR
from scipy.stats import zscore
df_zscore = zscore(nc as array, axis=1)
# dfz = pd.DataFrame(data)

# # Apply zscore to each column
# zscores_df = dfz.apply(zscore)

# # Print the DataFrame with Z-scores
# print("DataFrame with Z-scores:")
# print(zscores_df)

# Create a new DataFrame to store Z-scores
zscores_df = pd.DataFrame()

# Calculate Z-scores for each column and add to the new DataFrame
for column in dfz.columns:
    zscores_df[column] = zscore(dfz[column])


# Print the DataFrame with Z-scores
print("DataFrame with Z-scores:")
print(zscores_df)
### Skewness
from scipy.stats import skew

for col in nc:
    skewness = skew(df[col])
    print(f"Skewness of {col}: {skewness}")
## Data Transformation
df['log_attendence'] = np.log1p(df['Attendance'])
df

df['sqrt_attendence'] = np.sqrt(df['Attendance'])
df
df['cbrt_Attendance'] = np.cbrt(df['Attendance'])
df
## Insert New Column
df.shape

categories = ['A', 'B', 'C', 'D']
df['Div'] = np.random.choice(categories, size=len(df))
df

df['Div'] = df['Div'].fillna(df['Div'].mode())
df

encoded_df = pd.get_dummies(df['Div'],dtype=int)
df = pd.concat([df, encoded_df], axis=1)
df
