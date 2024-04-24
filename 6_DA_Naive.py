## ANALYTICS III Iris Dataset - Naive Bayes Theorem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
## Load dataset
iris = load_iris(as_frame = True)
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.DataFrame(iris['target'], columns=['target'])
x.head()
y.head()
x.hist()
y.hist()
## Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=133)
## Feature Scaling
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
std_scaler.fit_transform(x_train,x_test)
## Data is distributed normally hence Gaussian Dataset
from sklearn.naive_bayes import GaussianNB 
model = GaussianNB()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
y_predict
#y_test
# for i in range(len(y_predict)):
#       print(y_test[i],y_predict[i])
## Confusion Matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
cm = confusion_matrix(y_test,y_predict)
cm_d = ConfusionMatrixDisplay(cm).plot()
## Accuracy, Precision
from sklearn.metrics import accuracy_score,precision_score,recall_score
acc_score = accuracy_score(y_test,y_predict)
acc_score
prec_score = precision_score(y_test,y_predict,average='macro')
prec_score
## User Input
X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
prediction = model.predict(X_new)
print("Prediction of Species: {}".format(prediction))
new_data = np.array([[2.5,3.6,4.2,1.2]])
new_data = std_scaler.fit_transform(new_data)
print(model.predict(new_data))
new_data = np.array([[2.5,3.6,4.2,1.2]])
s_ff = std_scaler.fit(x_train)
new_data = s_ff.transform(new_data)
print(model.predict(new_data))
new_data = np.array([[5.1,3.5,1.4,0.2]])
s_ff = std_scaler.fit(x_train)
new_data = s_ff.transform(new_data)
print(model.predict(new_data))
new_d = np.array([[2,3,6,1.2]])
new_d = std_scaler.fit_transform(new_d)
print(model.predict(new_d))
