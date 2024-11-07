import os
seed_value= 1
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

import pandas as pd
import re
import matplotlib.pyplot as plt

#import seaborn as sns
import time

import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


import warnings
from sklearn.cluster import KMeans



data = pd.read_csv('/content/status col cell pellet M1 positive.csv')

display(data)

class_data = data[['Status']]

data2 = data.drop(['Compound IDs','Status'],axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label= le.fit_transform(class_data)

label2= pd.DataFrame(label)
label2.columns=['Status']

len(label)

X= data2
y= label2

display(X,y)

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,VotingClassifier
from sklearn.model_selection import KFold,GroupKFold, StratifiedKFold,train_test_split,RepeatedStratifiedKFold,LeaveOneOut
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#chi2 LOO

SEEDS= [0,44]

features=[]
features1=[]
cv_score =[]
pr_score =[]
re_score= []
au_score =[]
y_test=[]
y_pred=[]
model_rf=RandomForestClassifier(n_estimators=100, class_weight='balanced',oob_score=False,random_state=44)

for seed in SEEDS:
    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,random_state=seed)
    #KFold(n_splits=14)
    #

    for train_index, test_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]

        y_train, y_val = y.iloc[train_index], y.iloc[test_index]



        X_train1 = MinMaxScaler().fit_transform(X_train)
        X_train2= pd.DataFrame(X_train1)
        col_name1=X_train.columns
        X_train2.columns=col_name1

        X_val1 = MinMaxScaler().fit_transform(X_val)
        X_val2= pd.DataFrame(X_val1)
        X_val2.columns=col_name1

        smo = SMOTE(k_neighbors=1,random_state=42)
        X_train2_sm, y_train_sm = smo.fit_resample(X_train2, y_train)

        kf2 = LeaveOneOut()

        for train_index2, test_index2 in kf2.split(X_train2_sm, y_train_sm):
            X_train21, X_val21 = X_train2_sm.iloc[train_index2], X_train2_sm.iloc[test_index2]
            y_train21, y_val21 = y_train_sm.iloc[train_index2], y_train_sm.iloc[test_index2]

        #X_val1 = StandardScaler().fit_transform(X_val)
            sel= SelectKBest(chi2, k=10)
            sel.fit(X_train21, y_train21)
            sel1 = sel.get_support()
            feature = X_train21.iloc[:,sel1].columns.tolist()


        X_train22= X_train2_sm[feature]
        X_val22= X_val2[feature]



        model_rf.fit(X_train22, y_train_sm)


        ypred=model_rf.predict(X_val22)
        ypred_proba=model_rf.predict_proba(X_val22)
        #print(ypred.shape)
        #print(X_val22.shape)


        chi2tn, chi2fp, chi2fn, chi2tp = confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        chi2cm=confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        chi2total=sum(chi2cm)
        chi2accuracy=(chi2tp+chi2tn)/(chi2tn + chi2fp + chi2fn + chi2tp)
        #sensitivity = tp/(tp+fn)
        #specificity = tn/(tn+fp)

        #auc= roc_auc_score(y_val, model_rf.predict_proba(X_val22)[:, 1])


        cv_score.append(chi2accuracy)
        #pr_score.append(specificity)
        #re_score.append(sensitivity)

        y_test.extend(y_val['Status'].values)
        y_pred.extend(ypred)




        features.append(feature)
        features1.extend(feature)

'''

print('\nMean Accuracy',np.mean(cv_score))
chi2tn, chi2fp, chi2fn, chi2tp = confusion_matrix(y_test, y_pred,labels=[0,1]).ravel()
sensitivity = chi2tp/(chi2tp+chi2fn)
specificity = chi2tn/(chi2tn+chi2fp)
print('\nMean Specificity',specificity)
print('\nMean Sensitivity',sensitivity)

'''

cfeatures12= pd.DataFrame(features1)
cfeatures12.columns=["features"]
cfeature_count=cfeatures12.groupby('features').value_counts()
cfeature_count2= cfeature_count.to_frame()
cfeature_count2.columns=["count"]
cfeature_count2 = cfeature_count2.reset_index()
cfeature_count2= cfeature_count2.sort_values('count',ascending=False)
cfeature_count2

#LOG REGRESSION LOO

from sklearn.linear_model import LogisticRegression

SEEDS= [0,44]

features=[]
features1=[]
cv_score =[]
pr_score =[]
re_score= []
au_score =[]
y_test=[]
y_pred=[]
model_rf=RandomForestClassifier(n_estimators=100, class_weight='balanced',oob_score=False,random_state=44)

for seed in SEEDS:
    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,random_state=seed)
    #KFold(n_splits=14)
    #

    for train_index, test_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]

        y_train, y_val = y.iloc[train_index], y.iloc[test_index]



        X_train1 = MinMaxScaler().fit_transform(X_train)
        X_train2= pd.DataFrame(X_train1)
        col_name1=X_train.columns
        X_train2.columns=col_name1

        X_val1 = MinMaxScaler().fit_transform(X_val)
        X_val2= pd.DataFrame(X_val1)
        X_val2.columns=col_name1

        smo = SMOTE(k_neighbors=1,random_state=42)
        X_train2_sm, y_train_sm = smo.fit_resample(X_train2, y_train)

        kf2 = LeaveOneOut()

        for train_index2, test_index2 in kf2.split(X_train2_sm, y_train_sm):
            X_train21, X_val21 = X_train2_sm.iloc[train_index2], X_train2_sm.iloc[test_index2]
            y_train21, y_val21 = y_train_sm.iloc[train_index2], y_train_sm.iloc[test_index2]

        #X_val1 = StandardScaler().fit_transform(X_val)
            sel= SelectFromModel(LogisticRegression(penalty="l2",random_state=4), threshold='mean',max_features=10)
            sel.fit(X_train21, y_train21)
            sel1 = sel.get_support()
            feature = X_train21.iloc[:,sel1].columns.tolist()


        X_train22= X_train2_sm[feature]
        X_val22= X_val2[feature]



        model_rf.fit(X_train22, y_train_sm)


        ypred=model_rf.predict(X_val22)
        ypred_proba=model_rf.predict_proba(X_val22)
        #print(ypred.shape)
        #print(X_val22.shape)


        logtn, logfp, logfn, logtp = confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        logcm=confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        logtotal=sum(logcm)
        logaccuracy=(logtp+logtn)/(logtp+logtn+logfn+logfp)
        #sensitivity = tp/(tp+fn)
        #specificity = tn/(tn+fp)

        #auc= roc_auc_score(y_val, model_rf.predict_proba(X_val22)[:, 1])

        #print(accuracy)
        cv_score.append(logaccuracy)
        #pr_score.append(specificity)
        #re_score.append(sensitivity)

        y_test.extend(y_val['Status'].values)
        y_pred.extend(ypred)




        features.append(feature)
        features1.extend(feature)

'''

print('\nMean Accuracy',np.mean(cv_score))
logtn, logfp, logfn, logtp = confusion_matrix(y_test, y_pred,labels=[0,1]).ravel()
sensitivity = logtp/(logtp+logfn)
specificity = logtn/(logtn+logfp)
print('\nMean Specificity',specificity)
print('\nMean Sensitivity',sensitivity)

'''

efeatures12= pd.DataFrame(features1)
efeatures12.columns=["features"]
efeature_count=efeatures12.groupby('features').value_counts()
efeature_count2= efeature_count.to_frame()
efeature_count2.columns=["count"]
efeature_count2 = efeature_count2.reset_index()
efeature_count2= efeature_count2.sort_values('count',ascending=False)
efeature_count2

#ANOVA F VALUE LOO

from sklearn.feature_selection import f_classif
SEEDS= [0,44]

features=[]
features1=[]
cv_score =[]
pr_score =[]
re_score= []
au_score =[]
y_test=[]
y_pred=[]
model_rf=RandomForestClassifier(n_estimators=100, class_weight='balanced',oob_score=False,random_state=44)

for seed in SEEDS:
    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,random_state=seed)
    #KFold(n_splits=14)
    #

    for train_index, test_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]

        y_train, y_val = y.iloc[train_index], y.iloc[test_index]



        X_train1 = MinMaxScaler().fit_transform(X_train)
        X_train2= pd.DataFrame(X_train1)
        col_name1=X_train.columns
        X_train2.columns=col_name1

        X_val1 = MinMaxScaler().fit_transform(X_val)
        X_val2= pd.DataFrame(X_val1)
        X_val2.columns=col_name1

        smo = SMOTE(k_neighbors=1,random_state=42)
        X_train2_sm, y_train_sm = smo.fit_resample(X_train2, y_train)

        kf2 = LeaveOneOut()

        for train_index2, test_index2 in kf2.split(X_train2_sm, y_train_sm):
            X_train21, X_val21 = X_train2_sm.iloc[train_index2], X_train2_sm.iloc[test_index2]
            y_train21, y_val21 = y_train_sm.iloc[train_index2], y_train_sm.iloc[test_index2]

        #X_val1 = StandardScaler().fit_transform(X_val)
            sel= SelectKBest(f_classif, k=10)
            sel.fit(X_train21, y_train21)
            sel1 = sel.get_support()
            feature = X_train21.iloc[:,sel1].columns.tolist()


        X_train22= X_train2_sm[feature]
        X_val22= X_val2[feature]



        model_rf.fit(X_train22, y_train_sm)


        ypred=model_rf.predict(X_val22)
        ypred_proba=model_rf.predict_proba(X_val22)
        #print(ypred.shape)
        #print(X_val22.shape)


        ANOVAtn, ANOVAfp, ANOVAfn, ANOVAtp = confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        ANOVAcm=confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        ANOVAtotal=sum(ANOVAcm)
        ANOVAaccuracy=(ANOVAtp+ANOVAtn)/(ANOVAtp+ANOVAtn+ANOVAfn+ANOVAfp)
        #sensitivity = tp/(tp+fn)
        #specificity = tn/(tn+fp)
        #print(accuracy)

        #auc= roc_auc_score(y_val, model_rf.predict_proba(X_val22)[:, 1])


        cv_score.append(ANOVAaccuracy)
        #pr_score.append(specificity)
        #re_score.append(sensitivity)

        y_test.extend(y_val['Status'].values)
        y_pred.extend(ypred)




        features.append(feature)
        features1.extend(feature)

'''

print('\nMean Accuracy',np.mean(cv_score))
ANOVAtn, ANOVAfp, ANOVAfn, ANOVAtp = confusion_matrix(y_test, y_pred,labels=[0,1]).ravel()
sensitivity = ANOVAtp/(ANOVAtp+ANOVAfn)
specificity = ANOVAtn/(ANOVAtn+ANOVAfp)
print('\nMean Specificity',specificity)
print('\nMean Sensitivity',sensitivity)

'''

dfeatures12= pd.DataFrame(features1)
dfeatures12.columns=["features"]
dfeature_count=dfeatures12.groupby('features').value_counts()
dfeature_count2= dfeature_count.to_frame()
dfeature_count2.columns=["count"]
dfeature_count2 = dfeature_count2.reset_index()
dfeature_count2= dfeature_count2.sort_values('count',ascending=False)
dfeature_count2

all_features=pd.concat([cfeatures12,dfeatures12,efeatures12])

all_feature_count=all_features.groupby('features').value_counts()
all_feature_count2= all_feature_count.to_frame()
all_feature_count2.columns=["count"]
all_feature_count2 = all_feature_count2.reset_index()
all_feature_count2= all_feature_count2.sort_values('count',ascending=False)
all_feature_count2


#Models without feature selection

#plain RF

SEEDS= [0,44]

plaincv_score =[]
plainpr_score =[]
plainre_score= []
plainf1_score =[]
plainy_test=[]
plainy_pred=[]
model_rf=RandomForestClassifier(n_estimators=100, class_weight='balanced',oob_score=False,random_state=44)

for seed in SEEDS:
    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,random_state=seed)
    #KFold(n_splits=14)
    #

    for train_index, test_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]

        y_train, y_val = y.iloc[train_index], y.iloc[test_index]



        X_train1 = MinMaxScaler().fit_transform(X_train)
        X_train2= pd.DataFrame(X_train1)
        col_name1=X_train.columns
        X_train2.columns=col_name1

        X_val1 = MinMaxScaler().fit_transform(X_val)
        X_val2= pd.DataFrame(X_val1)
        X_val2.columns=col_name1

        smo = SMOTE(k_neighbors=1,random_state=42)
        X_train2_sm, y_train_sm = smo.fit_resample(X_train2, y_train)

        model_rf.fit(X_train2_sm, y_train_sm)


        ypred=model_rf.predict(X_val2)
        ypred_proba=model_rf.predict_proba(X_val2)
        #print(ypred.shape)
        #print(X_val22.shape)


        plaintn, plainfp, plainfn, plaintp = confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        plaincm=confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        plaintotal=sum(plaincm)
        plainaccuracy=(plaintp+plaintn)/(plaintp+plaintn+plainfn+plainfp)
        plainrecall = plaintp/(plaintp+plainfn)
        plainprecision = plaintn/(plaintn+plainfp)
        plainf1 = 2*(plainprecision*plainrecall)/(plainprecision+plainrecall)


        plaincv_score.append(plainaccuracy)
        plainpr_score.append(plainprecision)
        plainre_score.append(plainrecall)
        plainf1_score.append(plainf1)

        plainy_test.extend(y_val['Status'].values)
        plainy_pred.extend(ypred)

'''

print('\nMean Accuracy',np.mean(plaincv_score))
print('\nMean Precision',np.mean(plainpr_score))
print('\nMean Recall',np.mean(plainre_score))
print('\nMean F1 Score',np.mean(plainf1_score))

'''

#plain logreg

SEEDS= [0,44]

plain_logregcv_score =[]
plain_logregpr_score =[]
plain_logregre_score= []
plain_logregf1_score =[]
plain_logregy_test=[]
plain_logregy_pred=[]
model_logreg = LogisticRegression(penalty="l2",random_state=0)

for seed in SEEDS:
    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,random_state=seed)
    #KFold(n_splits=14)
    #

    for train_index, test_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]

        y_train, y_val = y.iloc[train_index], y.iloc[test_index]



        X_train1 = MinMaxScaler().fit_transform(X_train)
        X_train2= pd.DataFrame(X_train1)
        col_name1=X_train.columns
        X_train2.columns=col_name1

        X_val1 = MinMaxScaler().fit_transform(X_val)
        X_val2= pd.DataFrame(X_val1)
        X_val2.columns=col_name1

        smo = SMOTE(k_neighbors=1,random_state=42)
        X_train2_sm, y_train_sm = smo.fit_resample(X_train2, y_train)

        model_logreg.fit(X_train2_sm, y_train_sm)


        ypred=model_logreg.predict(X_val2)

        plain_logregtn, plain_logregfp, plain_logregfn, plain_logregtp = confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        plain_logregcm=confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        plain_logregtotal=sum(plain_logregcm)
        plain_logregaccuracy=(plain_logregtp+plain_logregtn)/(plain_logregtp+plain_logregtn+plain_logregfn+plain_logregfp)
        plain_logregrecall = plain_logregtp/(plain_logregtp+plain_logregfn)
        plain_logregprecision = plain_logregtn/(plain_logregtn+plain_logregfp)
        plain_logregf1 = 2*(plain_logregprecision*plain_logregrecall)/(plain_logregprecision+plain_logregrecall)


        plain_logregcv_score.append(plain_logregaccuracy)
        plain_logregpr_score.append(plain_logregprecision)
        plain_logregre_score.append(plain_logregrecall)
        plain_logregf1_score.append(plain_logregf1)

        plain_logregy_test.extend(y_val['Status'].values)
        plain_logregy_pred.extend(ypred)

'''

print('\nMean Accuracy',np.mean(plain_logregcv_score))
print('\nMean Precision',np.mean(plain_logregpr_score))
print('\nMean Recall',np.mean(plain_logregre_score))
print('\nMean F1 Score',np.mean(plain_logregf1_score))

'''

#plain SVM

SEEDS= [0,44]

plain_svmcv_score =[]
plain_svmpr_score =[]
plain_svmre_score= []
plain_svmf1_score =[]
plain_svmy_test=[]
plain_svmy_pred=[]
model_SVM = SVC(kernel='rbf')

for seed in SEEDS:
    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,random_state=seed)
    #KFold(n_splits=14)
    #

    for train_index, test_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]

        y_train, y_val = y.iloc[train_index], y.iloc[test_index]



        X_train1 = MinMaxScaler().fit_transform(X_train)
        X_train2= pd.DataFrame(X_train1)
        col_name1=X_train.columns
        X_train2.columns=col_name1

        X_val1 = MinMaxScaler().fit_transform(X_val)
        X_val2= pd.DataFrame(X_val1)
        X_val2.columns=col_name1

        smo = SMOTE(k_neighbors=1,random_state=42)
        X_train2_sm, y_train_sm = smo.fit_resample(X_train2, y_train)

        model_SVM.fit(X_train2_sm, y_train_sm)


        ypred=model_SVM.predict(X_val2)

        plain_svmtn, plain_svmfp, plain_svmfn, plain_svmtp = confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        plain_svmcm=confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        plain_svmtotal=sum(plain_svmcm)
        plain_svmaccuracy=(plain_svmtp+plain_svmtn)/(plain_svmtp+plain_svmtn+plain_svmfn+plain_svmfp)
        plain_svmrecall = plain_svmtp/(plain_svmtp+plain_svmfn)
        plain_svmprecision = plain_svmtn/(plain_svmtn+plain_svmfp)
        plain_svmf1 = 2*(plain_svmprecision*plain_svmrecall)/(plain_svmprecision+plain_svmrecall)


        plain_svmcv_score.append(plain_svmaccuracy)
        plain_svmpr_score.append(plain_svmprecision)
        plain_svmre_score.append(plain_svmrecall)
        plain_svmf1_score.append(plain_svmf1)

        plain_svmy_test.extend(y_val['Status'].values)
        plain_svmy_pred.extend(ypred)

'''

print('\nMean Accuracy',np.mean(plain_svmcv_score))
print('\nMean Precision',np.mean(plain_svmpr_score))
print('\nMean Recall',np.mean(plain_svmre_score))
print('\nMean F1 Score',np.mean(plain_svmf1_score))

'''

#selected features only RF

SEEDS= [0,44]

selcv_score =[]
selpr_score =[]
selre_score= []
self1_score =[]
sely_test=[]
sely_pred=[]

selected_features = all_features['features'].unique()

X_selected = X[selected_features]

y_selected = y

model_rf=RandomForestClassifier(n_estimators=100, class_weight='balanced',oob_score=False,random_state=44)

for seed in SEEDS:
    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,random_state=seed)
    #KFold(n_splits=14)
    #

    for train_index, test_index in kf.split(X_selected, y_selected):
        X_train, X_val = X_selected.iloc[train_index], X_selected.iloc[test_index]

        y_train, y_val = y_selected.iloc[train_index], y_selected.iloc[test_index]



        X_train1 = MinMaxScaler().fit_transform(X_train)
        X_train2= pd.DataFrame(X_train1)
        col_name1=X_train.columns
        X_train2.columns=col_name1

        X_val1 = MinMaxScaler().fit_transform(X_val)
        X_val2= pd.DataFrame(X_val1)
        X_val2.columns=col_name1

        smo = SMOTE(k_neighbors=1,random_state=42)
        X_train2_sm, y_train_sm = smo.fit_resample(X_train2, y_train)

        model_rf.fit(X_train2_sm, y_train_sm)


        ypred=model_rf.predict(X_val2)
        ypred_proba=model_rf.predict_proba(X_val2)
        #print(ypred.shape)
        #print(X_val22.shape)


        seltn, selfp, selfn, seltp = confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        selcm=confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        seltotal=sum(selcm)
        selaccuracy=(seltp+seltn)/(seltp+seltn+selfn+selfp)
        selrecall = seltp/(seltp+selfn)
        selprecision = seltn/(seltn+selfp)
        self1 = 2*(selprecision*selrecall)/(selprecision+selrecall)


        selcv_score.append(selaccuracy)
        selpr_score.append(selprecision)
        selre_score.append(selrecall)
        self1_score.append(self1)

        sely_test.extend(y_val['Status'].values)
        sely_pred.extend(ypred)

'''

print('\nMean Accuracy',np.mean(selcv_score))
print('\nMean Precision',np.mean(selpr_score))
print('\nMean Recall',np.mean(selre_score))
print('\nMean F1 Score',np.mean(self1_score))

'''

#selected features only logreg

SEEDS= [0,44]

lrcv_score =[]
lrpr_score =[]
lrre_score= []
lrf1_score =[]
lry_test=[]
lry_pred=[]

# Step 1: Extract the unique features selected across all three methods
selected_features = all_features['features'].unique()

# Step 2: Filter the original dataset to include only these features
X_selected = X[selected_features]

# Ensure your target labels `y` are in the right format
y_selected = y

model_logreg = LogisticRegression(penalty="l2",random_state=0)

for seed in SEEDS:
    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,random_state=seed)
    #KFold(n_splits=14)
    #

    for train_index, test_index in kf.split(X_selected, y_selected):
        X_train, X_val = X_selected.iloc[train_index], X_selected.iloc[test_index]

        y_train, y_val = y_selected.iloc[train_index], y_selected.iloc[test_index]



        X_train1 = MinMaxScaler().fit_transform(X_train)
        X_train2= pd.DataFrame(X_train1)
        col_name1=X_train.columns
        X_train2.columns=col_name1

        X_val1 = MinMaxScaler().fit_transform(X_val)
        X_val2= pd.DataFrame(X_val1)
        X_val2.columns=col_name1

        smo = SMOTE(k_neighbors=1,random_state=42)
        X_train2_sm, y_train_sm = smo.fit_resample(X_train2, y_train)

        model_logreg.fit(X_train2_sm, y_train_sm)


        ypred=model_logreg.predict(X_val2)
        ypred_proba=model_logreg.predict_proba(X_val2)
        #print(ypred.shape)
        #print(X_val22.shape)


        lrtn, lrfp, lrfn, lrtp = confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        lrcm=confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        lrtotal=sum(lrcm)
        lraccuracy=(lrtp+lrtn)/(lrtp+lrtn+lrfn+lrfp)
        lrrecall = lrtp/(lrtp+lrfn)
        lrprecision = lrtn/(lrtn+lrfp)
        lrf1 = 2*(lrprecision*lrrecall)/(lrprecision+lrrecall)


        lrcv_score.append(lraccuracy)
        lrpr_score.append(lrprecision)
        lrre_score.append(lrrecall)
        lrf1_score.append(lrf1)

        lry_test.extend(y_val['Status'].values)
        lry_pred.extend(ypred)

'''

print('\nMean Accuracy',np.mean(lrcv_score))
print('\nMean Precision',np.mean(lrpr_score))
print('\nMean Recall',np.mean(lrre_score))
print('\nMean F1 Score',np.mean(lrf1_score))

'''

#selected features only SVM

SEEDS= [0,44]

svmcv_score = []
svmpr_score =[]
svmre_score= []
svmf1_score =[]
svmy_test=[]
svmy_pred=[]

# Step 1: Extract the unique features selected across all three methods
selected_features = all_features['features'].unique()

# Step 2: Filter the original dataset to include only these features
X_selected = X[selected_features]

# Ensure your target labels `y` are in the right format
y_selected = y

model_SVM = SVC(kernel='rbf')

for seed in SEEDS:
    kf = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,random_state=seed)
    #KFold(n_splits=14)
    #

    for train_index, test_index in kf.split(X_selected, y_selected):
        X_train, X_val = X_selected.iloc[train_index], X_selected.iloc[test_index]

        y_train, y_val = y_selected.iloc[train_index], y_selected.iloc[test_index]



        X_train1 = MinMaxScaler().fit_transform(X_train)
        X_train2= pd.DataFrame(X_train1)
        col_name1=X_train.columns
        X_train2.columns=col_name1

        X_val1 = MinMaxScaler().fit_transform(X_val)
        X_val2= pd.DataFrame(X_val1)
        X_val2.columns=col_name1

        smo = SMOTE(k_neighbors=1,random_state=42)
        X_train2_sm, y_train_sm = smo.fit_resample(X_train2, y_train)

        model_SVM.fit(X_train2_sm, y_train_sm)


        ypred=model_SVM.predict(X_val2)

        svmtn, svmfp, svmfn, svmtp = confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        svmcm=confusion_matrix(y_val, ypred,labels=[0,1]).ravel()
        svmtotal=sum(svmcm)
        svmaccuracy=(svmtp+svmtn)/(svmtp+svmtn+svmfn+svmfp)
        svmrecall = svmtp/(svmtp+svmfn)
        svmprecision = svmtn/(svmtn+svmfp)
        svmf1 = 2*(svmprecision*svmrecall)/(svmprecision+svmrecall)


        svmcv_score.append(svmaccuracy)
        svmpr_score.append(svmprecision)
        svmre_score.append(svmrecall)
        svmf1_score.append(svmf1)

        svmy_test.extend(y_val['Status'].values)
        svmy_pred.extend(ypred)

'''

print('\nMean Accuracy',np.mean(svmcv_score))
print('\nMean Precision',np.mean(svmpr_score))
print('\nMean Recall',np.mean(svmre_score))
print('\nMean F1 Score',np.mean(svmf1_score))

'''
