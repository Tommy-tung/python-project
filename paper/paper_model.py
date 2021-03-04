#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import stackprinter 
import matplotlib.pyplot as plt
import random 
import joblib
from paper_dataprepare import data_preparation 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
stackprinter.set_excepthook(style='darkbg2')


# In[71]:


class hybrid : 
    
    def __init__(self, x_train, y_train, x_test, y_test, cv, score) :
        
        self.x_train = x_train
        self.y_train = np.array(y_train).ravel()
        self.x_test = x_test
        self.y_test = np.array(y_test).ravel()
        self.score = score
        self.cv = cv
        
    def scale(self) : 
        
        scale = StandardScaler()
        #scale.fit(train)
        self.x_train = scale.fit_transform(self.x_train)
        self.x_test = scale.transform(self.x_test)
        
    def pca(self, n) : 
        self.pca = PCA(n_components = n)
        self.x_train_pca = self.pca.fit_transform(self.x_train)
        self.x_test_pca = self.pca.transform(self.x_test)
        
        
    def logistic(self, param_grid ) : 
        
        self.logistic = GridSearchCV(LogisticRegression(), param_grid , scoring = self.score, cv = self.cv)
        self.logistic.fit(self.x_train, self.y_train)
        print('Logistic training complete ', self.logistic.best_estimator_, self.logistic.best_score_ ,sep = '\n')
        print('===============================')
        
    def svm(self, param_grid) :
        
        self.svm = GridSearchCV(SVC(), param_grid , scoring = self.score, cv = self.cv)
        self.svm.fit(self.x_train, self.y_train)
        print('SVM training complete', self.svm.best_estimator_, self.svm.best_score_ ,sep = '\n')
        print('===============================')
    
    def rf(self, param_grid) :
        
        self.rf = GridSearchCV(RandomForestClassifier(), param_grid , scoring = self.score, cv = self.cv)
        self.rf.fit(self.x_train, self.y_train)
        print('RF training complete', self.rf.best_estimator_, self.rf.best_score_ ,sep = '\n')
        print('===============================')
        
    def mlp(self, param_grid) :
        
        self.mlp = GridSearchCV(MLPClassifier(), param_grid , scoring = self.score, cv = self.cv)
        self.mlp.fit(self.x_train, self.y_train)
        print('MLP training complete', self.mlp.best_estimator_,self.mlp.best_score_ , sep = '\n')
        print('===============================')
        
        
    def meta_data(self) : 
        
        logit_train = pd.DataFrame(self.logistic.predict_proba(self.x_train)[:,0])
        logit_test = pd.DataFrame(self.logistic.predict_proba(self.x_test)[:,0])
        svm_train = pd.DataFrame(self.svm.predict_proba(self.x_train)[:,0])
        svm_test = pd.DataFrame(self.svm.predict_proba(self.x_test)[:,0])
        rf_train = pd.DataFrame(self.rf.predict_proba(self.x_train)[:,0])
        rf_test = pd.DataFrame(self.rf.predict_proba(self.x_test)[:,0])
        mlp_train = pd.DataFrame(self.mlp.predict_proba(self.x_train)[:,0])
        mlp_test = pd.DataFrame(self.mlp.predict_proba(self.x_test)[:,0])
        self.meta_train = pd.concat([pd.DataFrame(self.x_train_pca), logit_train, svm_train, rf_train, mlp_train], 
                                    axis = 1, 
                                    ignore_index = True)
        self.meta_test = pd.concat([pd.DataFrame(self.x_test_pca), logit_test, svm_test, rf_test, mlp_test], 
                                   axis = 1, 
                                   ignore_index = True)
     
    def hybrid_model(self, param_grid_h_rf, param_grid_h_mlp, cv, score) : 
        
        self.hybrid_rf = GridSearchCV(RandomForestClassifier(), 
                                      param_grid_h_rf , 
                                      scoring = score, 
                                      cv = cv)
        self.hybrid_rf.fit(self.meta_train, self.y_train)
        print('hybrid RF training complete', self.hybrid_rf.best_estimator_,self.hybrid_rf.best_score_ , sep = '\n')
        print('===============================')
        self.hybrid_mlp = GridSearchCV(MLPClassifier(), param_grid_h_mlp , scoring = score, cv = cv)
        self.hybrid_mlp.fit(self.meta_train, self.y_train)
        print('hybrid MLP training complete', self.hybrid_mlp.best_estimator_,self.hybrid_mlp.best_score_ , sep = '\n')
        print('===============================')
    
    def roc_curve(self, model, label, color, linestyle, dataset) : 
        
        for clf, label, clr, ls in zip(model, label, color, linestyle) : 
            if dataset == 'train' : 
                if 'hybrid'  in label : 
                    y_pred = clf.predict_proba(self.meta_train)[:, 1]
                else : 
                    y_pred = clf.predict_proba(self.x_train)[:, 1]
                fpr, tpr, threshold = roc_curve(y_true = self.y_train, 
                                            y_score = y_pred)
                plt.title('ROC curve(train_dataset)')
            if dataset == 'test' : 
                if 'hybrid'  in label : 
                    y_pred = clf.predict_proba(self.meta_test)[:, 1]
                else : 
                    y_pred = clf.predict_proba(self.x_test)[:, 1]
                fpr, tpr, threshold = roc_curve(y_true = self.y_test, 
                                            y_score = y_pred)
                plt.title('ROC curve(test_dataset)')
            roc_auc = auc(x = fpr, y = tpr)
            plt.plot(fpr, tpr, color = clr, 
                     linestyle = ls, 
                     label = '%s (auc = %0.2f)' %(label, roc_auc))
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], 
                 color='navy', 
                 lw =2, 
                 linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        

    @staticmethod    
    def metric(data,true_value, model) : 
        
        cnf =  confusion_matrix(true_value, 
                                model.predict( data))
        acc =  accuracy_score(true_value, model.predict( data))
        prec = precision_score(true_value, model.predict( data))
        recall = recall_score(true_value, model.predict( data))
        f1 = f1_score(true_value, model.predict( data))
        return cnf, acc, prec, recall, f1
    
    @staticmethod
    def save_model(model, path,name) : 
        
        saving_path = '/Users/tommy84729/python/論文/' + name
        joblib.dump(model, saving_path)
        
        
        

