THE SPARKS FOUNDATION : DATA SCIENCE AND BUSINESS ANALYTICS INTERNSHIP(GRIPJANUARY2022)
TASK 2 : Prediction Using Unsupervised ML(Level - Beginner)¶
AUTHOR : KAUSTUBH WANI

import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data=pd.read_csv("IRIS.csv")

data.head()

data.tail()

data.info()

data.describe()

sns.pairplot(data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']], hue="Species",diag_kind="kde")

sns.barplot(data['Species'],data['PetalWidthCm'])

x=data.drop(['Species'], axis=1)

Label_Encode=LabelEncoder()
Y=data['Species']
Y=Label_Encode.fit_transform(Y)

x=np.array(x)

x

Y

x=data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
Y=data['Species']

from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test=train_test_split(x,Y,test_size=0.2,random_state=1)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=1)
lr.fit(x_train,Y_train)

from sklearn.metrics import classification_report,confusion_matrix
y_predict=lr.predict(x_test)
conf=confusion_matrix(Y_test,y_predict)

print(classification_report(Y_test,y_predict))

print(conf)

train_score_lr = str(lr.score(x_train,Y_train)*100)
test_score_lr = str(lr.score(x_test,Y_test)*100)
accu_score_lr=str(accuracy_score(Y_test,y_predict)*100)
print(f'Train Score : {train_score_lr[:5]}%\nTest Score : {test_score_lr[:5]}%\nAccuracy Score : {accu_score_lr[:5]}%')
