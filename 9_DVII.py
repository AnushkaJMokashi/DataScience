# Data Visualization 9
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = sns.load_dataset('titanic')
type(df)
df
df.drop('deck',axis=1,inplace=True)
df
 ## BOX PLOT
sns.boxplot(x='sex',y='age',data=df)
## Box plots visually
#show the distribution of numerical data and skewness through displaying the
#data quartiles (or percentiles) and averages.
#### __INFERENCE__
# * Minimum age is 0
# * 25th-50th percentile people are of age from 20-30
# * 50th-75th percentile people are of age from 30-40
# * above 75th percentile people are of age abve 40
# * With hue we can also check the age group of people survived and not survived
sns.boxplot(x='age',y='sex',data=df)
sns.boxplot(x='sex',y='age',hue='survived', data = df)
# __Inference:__
# In above chart ,in addition to the information about the age of each gender, you can also see the distribution of the passengers who survived. For instance, you can see that among the male passengers, on average more younger people survived as compared to the older ones.
# ### Inter Quartile Range
df_age_sex = df[['age','sex']]
df_age_sex
Q1_male = df_age_sex[df_age_sex['sex']=='male']['age'].quantile(0.25)
Q1_female=df_age_sex[df_age_sex['sex']=='female']['age'].quantile(0.25)
Q3_male = df_age_sex[df_age_sex['sex']=='male']['age'].quantile(0.75)
Q3_female =df_age_sex[df_age_sex['sex']=='female']['age'].quantile(0.75)
IQR = Q3_male-Q1_male
IQR
IQR = Q3_female-Q1_female
IQR
## Violin Plot
# The Violin Plot The violin plot is similar to the box plot, however, 
# the violin plot allows us to
# display all the components that actually correspond to the data point.

# The violinplot() function is
# used to plot the violin plot. Like the box plot, the first parameter is the categorical column, the
# second parameter is the numeric column while the third parameter is the dataset.
sns.violinplot(x='sex',y='age',data=df)
sns.violinplot(x='sex',y='age',hue= 'survived',data=df)
sns.violinplot(x='sex',y='age',hue= 'survived',split = True,data=df)
# __Inference:__ For instance, if you look at the bottom of
# the violin plot for the males who survived (left-orange),                                                                    
# you can see that it is thicker than the
# bottom of the violin plot for the males who didn't survive (left-blue). 

# This means that the number
# of young male passengers who survived is greater than the number of young male passengers
# who did not survive.
## STRIPPLOT
# Strip plot plots the data points in the form of strip, with which we can observe the density of the range.
sns.stripplot(x='sex',y='age',data=df)
sns.stripplot(x='age',y='sex',data=df)
sns.stripplot(x='sex',y='age',dodge = True,hue = 'survived',size = 3,palette='Set1',data=df)
## Swarm Plot
# The swarm plot is a combination of the strip and the violin plots.

# In the swarmplots, the points are adjusted in such a way that they don't overlap.
sns.swarmplot(x='sex',y='age',data=df)

sns.swarmplot(x='sex',y='age',dodge = True,hue = 'survived',size = 3,palette='Set1',data=df)
# if you look at the bottom of
# the swarm plot for the males who survived (red), you can see that it is thicker than the
# bottom of the violin plot for the males who didn't survive (blue).


