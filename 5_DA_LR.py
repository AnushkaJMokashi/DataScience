#  Data Analytics II - Logistic Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
df = pd.read_csv('Social_Network_Ads.csv')
df
df.hist()
df.isnull().sum()
df.dropna()
df['EstimatedSalary'].plot.box() ##whisker boxplot min median max Q1 Q2 Q3
df.drop('Gender',axis=1,inplace=True)

df.drop('User ID',axis=1,inplace=True)
x = df.drop('Purchased',axis=1)  #drop column 
y = df['Purchased']
x
y
x.shape
y.shape
## Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(x,y,test_size=0.25,random_state=0)
x_train.shape
x_test.shape
## Feature Scaling
from sklearn.preprocessing import  StandardScaler
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)
## Model Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,Y_train)
y_predict = model.predict(x_test)
y_predict
## Performance Evaluation
from sklearn.metrics import accuracy_score,precision_score,recall_score
a_score = accuracy_score(Y_test,y_predict)
a_score
pre_score = precision_score(Y_test,y_predict)
pre_score
rec_score= recall_score(Y_test,y_predict)
rec_score
## Confusion Matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cm = confusion_matrix(Y_test,y_predict)

cm
cm_d = ConfusionMatrixDisplay(cm).plot()
## Predict over new value
p_value = [(46,41000)]
new_predict  = model.predict(p_value)
new_predict
# for  in range(len(y_predict)):
#     print(Y_test[i],y_predict[i])

# for i,p  in y_predict:
#     print(Y_test[i],y_predict[p])
## Classfication Report
from sklearn.metrics import classification_report
report = classification_report(Y_test,y_predict)
print(report)
