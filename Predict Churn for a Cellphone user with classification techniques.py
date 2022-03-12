# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:52:56 2021

@author: Sara Morani
"""

#%%1. IMPORT PACKAGES

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from dmba import plotDecisionTree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


#%%2. GET

df_cell=pd.read_csv('data/cellphone.csv')
df_cell.info()
df_cell.head()
df_cell.shape #(2151, 11)

#%%3. READY
y= df_cell['Churn']
X = df_cell.drop(['Churn', 'DataPlan','RoamMins'], axis=1)
X.info()
X.describe()

#Dropped dataplan as only 17% of the customers who didn't have a data plan, had data usage. This is assumed to be the data usage of free bundles as it was less 1GB for all instances. 
#Dropped RoamMins as it improved the model's performance.  


#%%4. PARTition

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=11)
#40% data kept as training set, random state set as per roll no. 

#OPTIONAL: To check if equal percentage of response after split for both train and test data
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

# 0   0.78
# 1   0.22
# Name: Churn, dtype: float64
# 0   0.77
# 1   0.23
# Name: Churn, dtype: float64

#Almost same % for churn (1) for train and test data. 


#%%Assignment Task 1 - Develop the best k-Nearest Neighbors (kNN) classifier


#%%5. NORM FOR ALL IN 1
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)   #only X_train's mean and std used for both train & test data

#Normalize with Trained Scale (array output has to be changed to DataFrame)
XNtrain = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
XNtest = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

#OPTIONAL: To see scale used to normalize each feature in both partitions
pd.options.display.float_format = '{:.2f}'.format  
dScales = pd.DataFrame(scaler.mean_, index=X.columns, columns=['Mean'])
dScales['Std'] = scaler.scale_
print(dScales)

#%%6. LOOP RANGE OF K'S AND RECORD RESULTS

resList=[]
for k in range(1,20):
    knn_ph = KNeighborsClassifier(n_neighbors=k)
    knn_ph.fit(XNtrain,y_train)
    y_pred = knn_ph.predict(XNtest)
    acc = metrics.accuracy_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred, pos_label=1)
    resList.append([k,acc,rec])
    
colsRes = ['k','Accuracy','Recall_Pos']
results = pd.DataFrame(resList, columns=colsRes)
print(results)

#Used the loop to find the optimal k. Results show the best k @1 and 9 with the highest recall_post of 57%.Since k=9 has slightly better accuracy of 86%, we'll use k=9.

#%%7.1.1 TRAIN

#Train kNN Model on best k
knn9_ph=KNeighborsClassifier(n_neighbors=9)
knn9_ph.fit(XNtrain,y_train)

#%%8.1.1 TEST
y_knn9 = knn9_ph.predict(XNtest)

#%%9.1.1 LEARN

#9.1 Confusion Matrix
cm9 = metrics.confusion_matrix(y_test,y_knn9)  
cm9Plot = metrics.ConfusionMatrixDisplay(cm9)
p9 = cm9Plot.plot(cmap='YlOrRd')
p9.figure_.savefig('figures/CM_KNN9.png')


#Confusion matrix shows that we have accuracy of 85.8% (627+112)/(627+36+86+112), while a recall of 56.6% (112/112+86). Out of all the positive churn classes, the model predicted 56% correctly.
#Looking at the specificity of the model, (627/627+36), 94.57% of no churn cases were identified correctly.  

#9.2 Classification Report
creport9 = metrics.classification_report(y_test,y_knn9)
print(creport9)

#               precision    recall  f1-score   support

#            0       0.88      0.95      0.91       663
#            1       0.76      0.57      0.65       198

#     accuracy                           0.86       861
#    macro avg       0.82      0.76      0.78       861
# weighted avg       0.85      0.86      0.85       861


#88% and 76% customers were identified correctly who didn't cancel and who did cancel their services respectively. Recall is the ability of a classifier model to find all positive instances which is 95% for customers who haven't cancelled their service. While only 57% were predicted for positive churned customers. This means that 43% customers who had churned were not identified. This seems to be a poor classification as our priority is on minimizing the false negatives i.e. customers who cancelled their service but were not identified as churned. k=9 does not seem to be the best predictor for churned customers. 

#9.3 kappa score
kappa9 = metrics.cohen_kappa_score(y_test,y_knn9)
print(kappa9)
#0.5610410450567911
#kappa score tells how much better the classifier is performing over the performance of a classifier that simply guesses at random according to the frequency of each class. 
#56% score seems to be a moderate agreement. 

#%%TRY THE SECOND BEST K@1

#%%7.1.2 TRAIN
knn1_ph=KNeighborsClassifier(n_neighbors=1)
knn1_ph.fit(XNtrain,y_train)

#%%8.1.2 TEST
y_knn1 = knn1_ph.predict(XNtest)

#%%9.1.2 LEARN

#9.1.1 Confusion Matrix
cm1 = metrics.confusion_matrix(y_test,y_knn1)  
cm1Plot = metrics.ConfusionMatrixDisplay(cm1)
p1 = cm1Plot.plot(cmap='YlOrRd')
p1.figure_.savefig('figures/CM_KNN1.png')


#the only difference from k=3 is of 22 classes, which were previously predicted currently as non-churned customers but are now classified as churned customers.   

#9.1.2 Classification Report
creport1 = metrics.classification_report(y_test,y_knn1)
print(creport1)

#               precision    recall  f1-score   support

#            0       0.88      0.95      0.91       663
#            1       0.76      0.57      0.65       198

#     accuracy                           0.86       861
#    macro avg       0.82      0.76      0.78       861
# weighted avg       0.85      0.86      0.85       861

#9.1.3 kappa score
kappa1 = metrics.cohen_kappa_score(y_test,y_knn1)
print(kappa1)
# 0.5031259017024141

#No improvements seen in the classification report or kappa score either


#%%Assignment Task 2 - Develop the best Classification Tree (CART) model which should not exceed a depth of 5 levels

#%%7.2 Train CART Model

#CART FULL TREE
ctree_full = DecisionTreeClassifier(random_state=11)
ctree_full.fit(X_train, y_train)
plotDecisionTree(ctree_full, feature_names = X.columns)

param_grid = {'max_depth':[2,3,4,5],
              'min_samples_split':[5,10,15,20],
              'min_impurity_decrease':[0, 0.0005, 0.001, 0.05, 0.01]
}
#try all combinations in the search of best tree. 

gridSearch = GridSearchCV(ctree_full,
                          param_grid,
                          n_jobs=-1,
                          scoring='recall')

gridSearch.fit(X_train,y_train)
print("Best Recall:", gridSearch.best_score_)
#Best Recall: 0.631578947368421
print("Best parameters:", gridSearch.best_params_)
#Best parameters: {'max_depth': 5, 'min_impurity_decrease': 0, 'min_samples_split': 15}

#_ tells its an attribute of the object and not a function.

ctree_best = gridSearch.best_estimator_
plotDecisionTree(ctree_best, feature_names = X.columns)


#TRAIN CART MODEL with max depth = 5
ctree5_ph = DecisionTreeClassifier(random_state=11, max_depth=5)

ctree5_ph.fit(X_train, y_train)

plotDecisionTree(ctree5_ph,
                 feature_names = X.columns,
                 class_names=ctree5_ph.classes_)


#%%8.2 TEST CART MODEL
y_ctree5 = ctree5_ph.predict(X_test)


#%%9.2 LEARN

#9.2.1 Confusion Matrix
cm_CART = metrics.confusion_matrix(y_test, y_ctree5)
cmPlot_CART = metrics.ConfusionMatrixDisplay(cm_CART)
p_CART=cmPlot_CART.plot(cmap='YlOrRd')
p_CART.figure_.savefig('figures/CM_CART.png')


#9.2.2 Classification Report
creport_CART = metrics.classification_report(y_test, y_ctree5)
print(creport_CART)

#               precision    recall  f1-score   support

#            0       0.89      0.95      0.92       663
#            1       0.80      0.60      0.68       198

#     accuracy                           0.87       861
#    macro avg       0.84      0.78      0.80       861
# weighted avg       0.87      0.87      0.87       861

#9.2.3 Kappa score
kappa_5 = metrics.cohen_kappa_score(y_test,y_ctree5)
print(kappa_5)
#0.6042173357069429

#For non-churned customer, the recall is 95% and 89% accuracy. For churned customers, the model has accuracy of 80% and only 60% of the true values were predicted correctly. This shows that CART model has imporved the results slightly when compared to knn @9.



#%%Assignment Task 3 - Develop a Random Forest model

#%% 7.3 Train Random Forest
rf_ph = RandomForestClassifier(n_estimators=500,
                                 random_state=11)
rf_ph.fit(X_train,y_train)

importances = rf_ph.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_ph.estimators_], axis=0)

df = pd.DataFrame({'feature': X.columns, 'importance': importances, 'std': std})
df = df.sort_values('importance', ascending=False)
#creates dataframe for the importance of each feature and the standard deviation in its contribution. 
print(df)

#            feature  importance  std
# 4          DayMins        0.21 0.07
# 6    MonthlyCharge        0.17 0.07
# 3    CustServCalls        0.16 0.03
# 7       OverageFee        0.12 0.03
# 2        DataUsage        0.09 0.05
# 0     AccountWeeks        0.08 0.02
# 5         DayCalls        0.08 0.02
# 1  ContractRenewal        0.08 0.02  - least important


ax = df.plot(kind='barh', xerr='std', x='feature', legend=False)
ax.set_ylabel('')
plt.show()

#%%8.3 Test Random Forest as a predictive model
y_rf = rf_ph.predict(X_test)


#%%9.3 Learn random forest
#9.1 Confusion Matrix
cm_rf = metrics.confusion_matrix(y_test, y_rf)
cmPlot_rf = metrics.ConfusionMatrixDisplay(cm_rf)
p_rf=cmPlot_rf.plot(cmap='YlOrRd')
p_rf.figure_.savefig('figures/CM_rf.png')


#9.2 Classification Report
creport_rf = metrics.classification_report(y_test, y_rf)
print(creport_rf)

#               precision    recall  f1-score   support

#            0       0.91      0.96      0.93       663
#            1       0.82      0.68      0.75       198

#     accuracy                           0.89       861
#    macro avg       0.87      0.82      0.84       861
# weighted avg       0.89      0.89      0.89       861


#9.3 Kappa score
kappa_rf = metrics.cohen_kappa_score(y_test,y_rf)
print(kappa_rf)
# 0.678963110667996

#recall at 68% of random forest for churned customers. Overall accuracy of 89%

#%%Assignment Task 4 - Report submitted as pdf

