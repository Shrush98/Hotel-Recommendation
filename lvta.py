# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:25:02 2019

@author: Shrushti
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from IPython.display import display

#Reading Data
dataset = pd.read_csv('lvta.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 13].values

#Missing Data(For Data Manipulation)
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'Nan' , strategy = 'mean',axis = 0)
#imputer = imputer.fit(X[:, :-1])
#X[:, :-1] =imputer.transform(X[:, :-1])



#Encoding
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X=LabelEncoder()

X[:,0]=labelencoder_X.fit_transform(X[:,0])
X[:,4]=labelencoder_X.fit_transform(X[:,4])
X[:,5]=labelencoder_X.fit_transform(X[:,5])
X[:,14]=labelencoder_X.fit_transform(X[:,14])

onehotencoder = OneHotEncoder(categorical_features = None,handle_unknown='ignore',dtype=float, n_values=None, sparse=True)
X = onehotencoder.fit_transform(X).toarray()

X[:,6]=labelencoder_X.fit_transform(X[:,6])
X[:,7]=labelencoder_X.fit_transform(X[:,7])
X[:,8]=labelencoder_X.fit_transform(X[:,8])
X[:,9]=labelencoder_X.fit_transform(X[:,9])
X[:,10]=labelencoder_X.fit_transform(X[:,10])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=0)

# Feature Scaling Data
from sklearn.preprocessing import StandardScaler 
sc_X =StandardScaler() 
X_train =sc_X.fit_transform(X_train)
X_test =sc_X.fit_transform(X_test)


#Describing data
display(dataset.describe(include=[np.number]))
display(dataset.describe(exclude=[np.number]))


#------NAIVEBAYES-----------------
#implemeting Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predict
y_pred = classifier.predict(X_test)

#Confusion matrix 
from sklearn.metrics import confusion_matrix,accuracy_score
cm1 = confusion_matrix(y_test, y_pred)
print("Naive Bayes accuracy: {} %".format(accuracy_score(y_pred,y_test)* 100))


#KNN(K-nearest Neighbours)
# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred)
print("KNN accuracy: {}%".format(accuracy_score(y_pred,y_test)* 100))



#Hierarchial Clustering

# Using the dendogram to find the optimal number of clusters
#import scipy.cluster.hierarchy as sch
#dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
#plt.title('Dendogram')
#plt.xlabel('Users')
#plt.ylabel('Hotels')
#plt.show()
#print("Hierarchial Clustering")
# Fitting hierarchical clustering to the mall dataset
#from sklearn.cluster import AgglomerativeClustering
#hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
#y_hc = hc.fit_predict(X)


#Random_Forest

# Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred)
print("Random Forest accuracy: {}%".format(accuracy_score(y_pred,y_test)* 100))
























