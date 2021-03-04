#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd 
import numpy as np
from paper_dataprepare import data_preparation 
from paper_model import hybrid


# In[3]:


data = pd.read_excel('/Users/tommy84729/python/論文/data/EI.xlsx',
                     encoding = 'utfx-8').drop('利息保障倍數', axis = 'columns')
macro = pd.read_csv('/Users/tommy84729/python/論文/data/macro.csv',
                    encoding = 'big5', index_col = None).drop('年', axis = 'columns')


# In[16]:


name1 = ['存貨週轉率（次）', '當季季底P/B']
name2 = ['存貨週轉率（次）', '應收帳款週轉次數', '總資產週轉次數', '固定資產週轉次數', '總負債/總淨值','當季季底P/B','淨值']
num = 0.00001
path = '/Users/tommy84729/python/論文/'


# In[ ]:


model = data_preparation(data, macro)


# In[5]:


model.macro_pre()
model.distress_pre()
model.create_data()
model.creat_X_Y()
model.creat_train_test(0.3)
model.x_train, model.y_train = model.clean_data(name1, name2, num, model.x_train, model.y_train)


# In[9]:


model.x_train


# In[10]:


hybrid_model = hybrid(model.x_train, model.y_train, model.x_test, model.y_test, 3, 'f1_weighted')


# In[11]:


hybrid_model.scale()
hybrid_model.pca(10)


# In[12]:


param_grid_lg = [
        {
            'penalty' : [ 'none'],
            'solver' : ['lbfgs'],
            'class_weight': [ 'balanced'],
            'fit_intercept' : [True],
            'max_iter' : [10000]
        }
       ]
param_grid_svm = [
        {
            'C' : [50],#10,50,100],
            'kernel' : ['rbf'],
            'probability' : [True],
            'class_weight': [ {0:1,1:1.5},{0:1,1:2},{0:1,1:2.5} ],
            'random_state' : [10],
        }
       ]
param_grid_rf = [
        {
            'n_estimators' : [200],#,300,400,500],
            'criterion' : ['gini'],
            'class_weight' : ['balanced'],
            'class_weight': [ {0:1,1:1.5},{0:1,1:2} ,{0:1,1:2.5}, {0:1,1:3}],
            'max_depth' : [6]
        }
       ]
param_grid_mlp = [
        {
            'activation' : ['relu'],
            'solver' : ['adam'],
            'hidden_layer_sizes': [
             (64,32,),(128,64,),(256,128,)#,(128,64,32,),(64,32,16,)
             ],
            'learning_rate' : ['constant'],
            'learning_rate_init' : [0.001, 0.005],
            'random_state' : [1],
            'early_stopping' : [True]
        }
       ]


# In[13]:


hybrid_model.logistic(param_grid_lg)
hybrid_model.svm(param_grid_svm)
hybrid_model.rf(param_grid_rf)
hybrid_model.mlp(param_grid_mlp)


# In[17]:


hybrid_model.save_model(hybrid_model.logistic, path, 'logit')
hybrid_model.save_model(hybrid_model.svm, path, 'svm')
hybrid_model.save_model(hybrid_model.rf, path, 'rf')
hybrid_model.save_model(hybrid_model.mlp, path, 'mlp')


# In[20]:


param_grid_h_rf = [
        {
            'n_estimators' : [1000],
            'criterion' : ['entropy'],
            #'class_weight': [{0:1,1:4}, {0:1,1:3} ],
            'max_depth' : [6]
            
        }
       ]

param_grid_h_mlp = [
        {
            'activation' : ['relu'],
            'solver' : ['adam'],
            'hidden_layer_sizes': [
             (64,32,),(128,64,)#,(128,64,32,),(32,16,)
             ],
            'learning_rate' : ['constant','adaptive'],
            'learning_rate_init' : [0.001, 0.005],
            'random_state' : [1],
            'max_iter' : [10000]
        }
       ]


# In[21]:


hybrid_model.meta_data()


# In[22]:


hybrid_model.hybrid_model(param_grid_h_rf, param_grid_h_mlp, 3, 'f1_weighted')


# In[23]:


clf_h = [hybrid_model.hybrid_rf, hybrid_model.hybrid_mlp, hybrid_model.rf, hybrid_model.mlp]
label_h = ['hybrid_RF', 'hybrid_MLP', 'RF', 'MLP']
color_h = [ 'orange', 'yellow', 'black', 'green']
ls_h = [':', '--', '-.', '-']


# In[24]:


hybrid_model.roc_curve(clf_h, label_h, color_h, ls_h,'test')


# In[ ]:




