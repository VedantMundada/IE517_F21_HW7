# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:58:36 2021

@author: Vedant
"""
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
path = "C:/Users/Vedant/Desktop/UIUC/SEM 1/Machine Learning/Week 6/ccdefault.csv"
data=pd.read_csv(path)
print(data.head())
data = data.drop('ID',axis=1)
df=data
print(data.head())
data=data.values

X = data[:,:22]
y = data[:,23]

#%%
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=1)
#%%
start = timeit.default_timer()

from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state=10 , max_depth=None)
tree.fit(X,y)

acc_tree=tree.score(X_test,y_test)
print("Accuracy (out f sample) for single decision tree = ",acc_tree)

stop = timeit.default_timer()
print('Time: ', stop - start,"secs") 
#%%
start = timeit.default_timer()

acc_bag=[]
acc_bag_train=[]
for i in range(1,6):
    bgc = BaggingClassifier(base_estimator=tree,n_estimators=i*100,
                            max_samples=1.0,max_features=1.0,bootstrap=True,bootstrap_features=False,
                            n_jobs=-1,random_state=10)
    bgc.fit(X_train,y_train)
    acc_bag_train.append(bgc.score(X_train,y_train))
    acc_bag.append(bgc.score(X_test,y_test))
    
plt.plot(acc_bag,label='Out of sample accuracy')
plt.plot(acc_bag_train,label='In sample accuracy')
plt.legend(loc='center')
plt.title("Accuracy scores (Bagging Tree)")
plt.xlabel('N Estimators (*100)')
plt.ylabel('Accuracy')
plt.show()


stop = timeit.default_timer()
print('Time: ', stop - start,"secs") 

#%%

start = timeit.default_timer()
from sklearn.ensemble import RandomForestClassifier
rfc_score=[]
rfc_score_test=[]
for i in range(1,6):
    rfc = RandomForestClassifier(n_estimators =100*i, criterion = 'entropy', max_depth=None,n_jobs=-1,random_state=10)
    rfc.fit(X_train,y_train)
    rfc_score.append(rfc.score(X_train,y_train))
    rfc_score_test.append(rfc.score(X_test,y_test))
    #print(rfc.score(X_test,y_test))
    
    
plt.plot(rfc_score_test,label='Out of sample accuracy')
plt.plot(rfc_score,label='In sample accuracy')
plt.legend(loc='center')
plt.title("Accuracy scores (Random FOrest)")
plt.xlabel('n_estimators (*100)')
plt.ylabel('Accuracy')
plt.show()
stop = timeit.default_timer()
print('Time: ', stop - start,"secs") 
#%%
start = timeit.default_timer()
index = rfc_score_test.index(max(rfc_score_test))
rfc2 = RandomForestClassifier(n_estimators =100*index, criterion = 'entropy', max_depth=None,n_jobs=-1,random_state=10)
acc_train=rfc2.fit(X_train,y_train)
acc_test = rfc.score(X_test,y_test)
stop = timeit.default_timer()
print('Time: ', stop - start,"secs") 
#%%
start = timeit.default_timer()
feat_labels = df.columns[0:]
importances = rfc2.feature_importances_
indices=np.argsort(importances)[::-1]
for i in range (X_train.shape[1]):
    print(i+1,"   ",feat_labels[indices[i]],"   ",importances[indices[i]])
    
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]),importances[indices],align = 'center')
plt.xticks(range(0,22),feat_labels[indices],rotation = 90)
plt.xlim([-1,X_train.shape[1]])
plt.show()

stop = timeit.default_timer()
print('Time: ', stop - start,"secs") 

#%%
plt.plot(acc_bag,label='Bagging Accuracy')
plt.plot(rfc_score_test,label="Random Forest Accuracies")   
plt.legend(loc='center')
plt.title("Bagging vs Random Forest")
plt.xlabel('n_estimators (*100)')
plt.ylabel('Accuracy')
plt.show()























