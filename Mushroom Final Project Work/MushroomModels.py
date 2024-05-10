# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:25:11 2023

@author: cpdeb
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 
from sklearn import linear_model, tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("agaricus-lepiota.data", names = ["Edibility","Cap-Shape","Cap-Surface","Cap-Color","Bruises","Odor","Gill-Attachment","Gill-spacing","Gill-Size","Gill-Color","Stalk-Shape","Stalk-Root","Stalk-Surface-Above","Stalk-Surface-Below","Stalk-Color-Above","Stalk-Color-Below","Veil-Type","Veil-Color","Ring-Number", "Ring-Type", "Spore-Print-Color","Population","Habitat"], index_col=False)

#Encode using ordinal enoder

df2 = df.copy()
encoder = OrdinalEncoder()
for column in df2:
    temp_column = encoder.fit_transform(df2[[column]])
    df2[column] = temp_column
    
df2 = df2.drop(columns=['Veil-Type'])
correlations = df2.corr()


x = df2[df2.columns[2:]]
y = df2["Edibility"]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=.7)

reg = linear_model.LogisticRegression()
reg_parameters = {'fit_intercept':[True,False], 'C':[1, 10, 15], 'penalty': [None, 'l1','l2']}

svm = svm.SVC()
svm_parameters = {'kernel':['rbf', 'linear', 'poly'], 'C':[1, 10, 15], 'degree':[1,3,5,10]}

kNN = KNeighborsClassifier()
kNN_parameters = {"n_neighbors":[5,10,20], "weights":['uniform','distance']}

tree = tree.DecisionTreeClassifier()
tree_parameters = {'max_depth':[2, 5, 10], 'min_samples_leaf':[5, 15, 30, 50]}

gnb = GaussianNB()
gnb_parameters = {}


gs_reg = GridSearchCV(reg, reg_parameters)
gs_reg.fit(X_train, Y_train)
print(gs_reg.best_params_)
#Best one is fit_intercept: True, C:15, penalty:l2
gs_svm = GridSearchCV(svm, svm_parameters)
gs_svm.fit(X_train, Y_train)
print(gs_svm.best_params_)
# best one is kernel: poly c:1 degree:5
gs_kNN = GridSearchCV(kNN, kNN_parameters)
gs_kNN.fit(X_train, Y_train)
print(gs_kNN.best_params_)
#best is neighbors:10 weights:distance
gs_tree = GridSearchCV(tree, tree_parameters)
gs_tree.fit(X_train, Y_train)
print(gs_tree.best_params_)
#best is depth:10, min samples: 5
gs_gnb = GridSearchCV(gnb, gnb_parameters)
gs_gnb.fit(X_train, Y_train)
print(gs_gnb.best_params_)


gs_reg.fit(X_train,Y_train)
reg_pred = gs_reg.predict(X_test)
print("reg confusion matirx: " + str(confusion_matrix(Y_test, reg_pred)))
print("reg f1 score: " + str(f1_score(Y_test, reg_pred)))
print("reg accuracy: " + str(accuracy_score(Y_test, reg_pred)))

gs_svm.fit(X_train,Y_train)
svm_pred = gs_svm.predict(X_test)
print("svm confusion matirx: " + str(confusion_matrix(Y_test, svm_pred)))
print("svm f1 score: " + str(f1_score(Y_test, svm_pred)))
print("svm accuracy: " + str(accuracy_score(Y_test,svm_pred)))

gs_kNN.fit(X_train,Y_train)
kNN_pred = gs_kNN.predict(X_test)
print("knn confusion matirx: " + str(confusion_matrix(Y_test, kNN_pred)))
print("knn f1 score: " + str(f1_score(Y_test, kNN_pred)))
print("knn accuracy: " + str(accuracy_score(Y_test,kNN_pred)))

gs_tree.fit(X_train,Y_train)
tree_pred = gs_tree.predict(X_test)
print("dtc confusion matirx: " + str(confusion_matrix(Y_test, tree_pred)))
print("dtc f1 score: " + str(f1_score(Y_test, tree_pred)))
print("dtc accuracy: " + str(accuracy_score(Y_test,tree_pred)))

gs_gnb.fit(X_train,Y_train)
gnb_pred = gs_gnb.predict(X_test)
print("gnb confusion matirx: " + str(confusion_matrix(Y_test, gnb_pred)))
print("gnb f1 score: " + str(f1_score(Y_test, gnb_pred)))
print("gnb accuracy: " + str(accuracy_score(Y_test,gnb_pred)))

dtc = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf = 5)
dtc.fit(X_train, Y_train)
plt.figure(figsize=(18,18))
tree.plot_tree(dtc, fontsize = 7, feature_names = list(df2.columns[2:]))
plt.show()