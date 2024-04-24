# 8 - Data Visualization 
import numpy as np
import pandas as pd
import seaborn as sns
df=sns.load_dataset("titanic")
df.head()
df.isnull().sum()
df.shape
df.info()
# ## Sactter Plot
# To check Outliers in the data
# sns.scatterplot(data=df,x='age',y='fare')
# In above graph, we can observe the outliers of column 'Fare' with respect to the column 'Age'.
# sns.scatterplot(data=df,x='age',y='fare',hue='sex')
# In above graph, we can observe the outliers of column 'Fare' with respect to the column 'Age'. As hue is 'Gender' it is also segregated w.r.t to gender.
# ## Histplot
# Histograms are visualization tools that represent the distribution of a set of continuous data. In a
# histogram, the data is divided into a set of intervals or bins (usually on the x-axis) and the count of
# data points that fall into each bin corresponding to the height of the bar above that bin. These bins
# may or may not be equal in width but are adjacent (with no gaps).
sns.histplot(data=df,x='age')
# __Inference:__ Histplot is used for continuos data. Above graph gives count per age group.
# sns.histplot(data=df,x='age',hue='pclass')
# __Above graph gives count of Age and w.r.t pclass column__
# sns.displot(data=df,x='age',hue='sex',bins=30)
# __In above graph,Age wise count w.r.t gender__
# sns.displot(data=df,x='age',hue='sex',bins=30,col='embark_town')
# In above graph, we can see Age wise count w.r.t gender for each town.

# With this count we can have clear idea of the people from the different town
## Bar Plot
sns.barplot(x='sex',y='age',data=df)
# __Inference:__ Above graph gives the age with respect to gender.
# sns.barplot(x='pclass',y='embark_town',data=df,orient='h')
# __Above graph gives pclass and town wise relation__
sns.barplot(x='sex',y='age',data=df,ci=None)
# ## Count Plot
# The count plot is similar to the bar plot, however it displays the count of the categories in a
# specific column. For instance, if we want to count the number of males and women passenger we
# can do so using count plot as follows:
sns.countplot(x='pclass',hue='sex',data=df)
sns.countplot(x='pclass',hue='embark_town',data=df)
sns.countplot(x='pclass',hue='survived',data=df)
# * We can observe that people from pclass 1 have survived more then pclass 3
# * Death of pclass 3 people is more
# sns.countplot(x='survived',hue='embark_town',data=df)
# * For above graph, survival of people town wise.
# * It is town wise count.
# Catplot
sns.catplot(x='embarked',hue='survived',data=df,kind='count')
sns.catplot(x='embarked',hue='survived',data=df,kind='count',col='pclass')
sns.catplot(x='survived',hue='embarked',data=df,kind='count',col='pclass')
# Cat plot is used to plot different graphs.
# The kind parameter can be changed according to the  requirement
