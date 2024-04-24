## Assignment 3 Descriptive Statistics Adult data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
df = pd.read_csv("data.csv")
df
df.columns
df.shape
df.info()
df.isnull().sum()
df.describe()
df.age.min()
df['age'].max()
df.age.mean()
df.age.std()
df.age.median()
df['income'].unique()
df['income'].nunique()   ##count the unique
df.groupby(['income','age']).count()
## total 142 rows for group 
## Group By
df.groupby(['income','age']).min()
df.groupby(['income','age']).max()
# df.groupby(['income','age']).mean()
df.groupby("income")['age'].count()  ## age specified
df.groupby("income").count()  ## for all
df.groupby("income")['age'].min()
df.groupby("income")['age'].max()
df.groupby("income")['age'].mean()
df.groupby("income")['age'].median()
df.groupby("income")['age'].std()
df.groupby(["income","age"])['hours-per-week'].min()
#summary statistics of age grouped by gender
df.groupby("gender")["age"].describe()
df.groupby("marital-status")["age"].mean() 
df.groupby("marital-status")["age"].median()
#grouping can be done on multiple columns
# summary statistics of age grouped by gender & marital-status
df.groupby(["gender","marital-status"])["age"].std() 
#Count number of records by category
#The value_counts() method counts the number of records for each category in a column.
df["marital-status"].value_counts()
## Using User Defined functions:
income_less_than_50 = df[df["income"]=="<=50K"]
print("Less than 50K",income_less_than_50.head())
income_greater_than_50 = df[df["income"]==">50K"]
print("Greater than 50K",income_greater_than_50.head())
def display_statistics(income_data,income_class):
    column = ["age","fnlwgt","educational-num","capital-gain","capital-loss","hours-per-week"]
    print("Statistics for Income - ",income_class)
    print("------------------------------------------------------")
    print("Mean:")
    print(income_data[column].mean())
    print("\n")
    
    print("------------------------------------------------------")
    print("Median:")
    print(income_data[column].median())
    print("\n")
    
    print("------------------------------------------------------")
    print("Standard Deviation:")
    print(income_data[column].std())
    print("\n")
    
    print("25% Percentile:")
    print(income_data[column].quantile(0.25))
    print("\n")

    print("75% Percentile:")
    print(income_data[column].quantile(0.75))
    print("\n")
    
    print("Minimum:")
    print(income_data[column].min())
    print("\n")
    
    print("Maximum:")
    print(income_data[column].max())
    
    
display_statistics(income_less_than_50,"<=50K")
print("\n")

display_statistics(income_greater_than_50,">50K")
def calculate_mean(data):
    if len(data)==0:
        return 0
    m = sum(data)/len(data)
    return m

def calculate_std(data,mean):
    if len(data)<=1:
        return 0
    difference_squared = sum((x-mean)**2 for x in data)
    ans = (difference_squared/(len(data)-1))**0.5
    return ans

def calculate_percentile(data,percentile):
    sorted_data = sorted(data)
    index = int(percentile*len(data))
    percentile_result = sorted_data[index]
    return percentile_result

def display_stats(income_data,income_class):
    column = ["age","fnlwgt","educational-num","capital-gain","capital-loss","hours-per-week"]
    print(f"\n****************Statistics for {income_class}*********************")
    
    # Mean
    mean_values = [calculate_mean(income_data[col]) for col in column]
    print("Mean: ")
    print(pd.Series(mean_values, index=column))

    # Standard Deviation
    std_values = [calculate_std(income_data[col],mean_values[i]) for i, col in enumerate(column)]
    print("\nStandard Deviation")
    print(pd.Series(std_values, index=column))

    # Percentile
    percentiles = [0.25, 0.75]
    for percentile_value in percentiles:
        percentile_values = [calculate_percentile(income_data[col], percentile_value) for col in column]
        print(f"\n{int(percentile_value * 100)}th Percentile : ")
        print(pd.Series(percentile_values, index=column))
display_stats(income_less_than_50, '<= 50K')
display_stats(income_less_than_50, '>50K')

#########################################################################################
## Assignment 3 Descriptive Statistics Iris Data
import pandas as pd
import seaborn as sns
import numpy as np
iris_data = sns.load_dataset('iris')
iris_data
iris_data.shape
iris_data.species.unique()
iris_data.species.nunique()
iris_data.describe()
iris_data.groupby(['species']).count()
setosa_data = iris_data[iris_data['species'] == 'setosa']
setosa_data
versicolor_data = iris_data[iris_data['species'] == 'versicolor']
versicolor_data
virginica_data = iris_data[iris_data['species'] == 'virginica']
virginica_data.head()
species_data_g = iris_data.groupby('species')
setosa_data_g
setosa_data.describe()
versicolor_data.describe()
virginica_data.describe()
nc = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','species']
# species_data = ['setosa','versicolor','viginica']
nc = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
def species_stats(species_data,species_name):
        print("Species Name: {}".format(species_name))
        print("Mean:",species_data[nc].mean())
        print("Median:",species_data[nc].median())
        print("std:",species_data[nc].std())
        print("25% percentile:",species_data[nc].quantile(0.25))
        print("75% percentile:",species_data[nc].quantile(0.75))
        print("Min:",species_data[nc].min())
        print("Max:",species_data[nc].max())
species_data_names = ['setosa_data','viginica_data','versicolor_data']
for data in species_data_names:
    print("************** Species name {} *****************".format(data))
    species_stats(setosa_data,data)
    print("------------------------------------")
print(setosa_data.nunique())
def calculate_mean(data):
    if len(data)==0:
        return 0
    m = sum(data)/len(data)
    return m

def calculate_std(data,mean):
    if len(data)<=1:
        return 0
    difference_squared = sum((x-mean)**2 for x in data)
    ans = (difference_squared/(len(data)-1))**0.5
    return ans

def calculate_percentile(data,percentile):
    sorted_data = sorted(data)
    index = int(percentile*len(data))
    percentile_result = sorted_data[index]
    return percentile_result

def display_stats(species_data,species_name):
    column = nc
    print(f"\n****************Statistics for {species_name}*********************")
    
    # Mean
    mean_values = [calculate_mean(species_data[col]) for col in column]
    print("Mean: ")
    print(pd.Series(mean_values, index=column))

    # Standard Deviation
    std_values = [calculate_std(species_data[col],mean_values[i]) for i, col in enumerate(column)]
    print("\nStandard Deviation")
    print(pd.Series(std_values, index=column))

    # Percentile
    percentiles = [0.25, 0.75]
    for percentile_value in percentiles:
        percentile_values = [calculate_percentile(species_data[col], percentile_value) for col in column]
        print(f"\n{int(percentile_value * 100)}th Percentile : ")
        print(pd.Series(percentile_values, index=column))

display_stats(setosa_data, 'Iris-setosa')
display_stats(versicolor_data, 'Iris-versicolor')
display_stats(virginica_data, 'Iris-virginica')
## Group By

iris_data.groupby(["species"])["sepal_length"].mean()
iris_data.groupby(["species"])["sepal_length"].std()
iris_data.groupby(["species"])["sepal_length"].describe()
iris_data.groupby(["species"])["sepal_length"].quantile(q=0.25)
iris_data.groupby(["species"])["sepal_length"].quantile(q=0.75)
a=iris_data.groupby(["species"])["sepal_length"].mean()
print(a) 
b=iris_data.groupby(["species"])["sepal_length"].median() 
print(b)

list=[a,b] 
print(list)


