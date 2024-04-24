## Data Visualization - iris
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')

df.columns
sns.displot(x='sepal_length',data=df,hue='species')
sns.displot(x='sepal_length',data=df)
# __Inference__:
# * In the histogram for sepal length, we observe that the distribution is somewhat normal with a peak around 5.8 cm.
# * Most of the iris flowers have a sepal length ranging from approximately 4.5 cm to 7.0 cm. 
# * There are relatively fewer flowers with extremely short or long sepal lengths, as evidenced by the lower frequencies at the tails of the distribution

sns.histplot(x='sepal_width',data=df)
# __Inference:__
# * The histogram for sepal width shows a roughly normal distribution, albeit with some variation. 
# * The most common sepal width appears to be around 3.0 cm. 
# * There is a noticeable spread in sepal widths, ranging from approximately 2.0 cm to 4.5 cm. However, there's a slight skew towards higher sepal widths, as indicated by the slightly longer right tail of the distribution.
sns.histplot(x='sepal_width',data=df)
# __Inference__:
# Shows th distribution of the sepal_width of the iris flowers. COunt for sepal width 3.0 is highest. The has a little bit of skewnwss
sns.histplot(x='petal_length',data=df)
# __Inference:__
# * Data is not normally distributed
# * Some of the petal length are ranging from 1-2
# * Most data is in range of 3-7
sns.histplot(x='petal_width',data=df)
# * Data is not normally distributed
# * Some of the petal length are ranging from 0-0.5
# * Most data is in range of 1.0-2.5
