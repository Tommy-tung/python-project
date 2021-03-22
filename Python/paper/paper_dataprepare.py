#!/usr/bin/env python
# coding: utf-8

# In[635]:

import pandas as pd 
import numpy as np
import random 
from sklearn.model_selection import train_test_split

class data_preparation : 
    
    def __init__(self, data, macro) : 
        self.data = data
        self.macro = macro
        self.macro_all = []
        self.distress = []
        self.data_full = []

        
    def macro_pre(self) : 
        year = list(self.data['年'].unique())
        for i in range(len(self.macro.columns)) : 
            value = list()
            for j in range(len(year)) : 
                data1 = self.data[self.data['年'] == year[j]]
                value.extend([self.macro.iloc[j,i]]*len(data1))
            self.macro_all.append(value)
        macro_all = pd.DataFrame(self.macro_all).T
        macro_all.columns = list(self.macro.columns)
        self.macro_all = macro_all
        #return macro_all
        
    def distress_pre(self) : 
        table = self.data.pivot(index = '公司', 
                                columns = '年', 
                                values = '常續性稅後淨利')
        for i in range(len(table)) :
            for j in range(len(table.columns) - 3) : 
                self.distress.append(table.iloc[i, j:j+4].tolist())
        self.distress= pd.DataFrame(self.distress)
        
        
        
    ##使用 3年平均值產生資料，總共有16個區間，32個變數
    def create_data(self) :
        
        df = pd.concat([self.data, self.macro_all], axis = 1).drop('常續性稅後淨利', axis = 1)
        for var in df.columns[3:] : 
            table = df.pivot(index = '公司', 
                             columns = '年',
                             values = var).rolling(3, min_periods = 3, axis = 1).mean()
            mean = []
            period = []
            company = []
            for i in range(len(table)) : 
                for j in range(len(table.columns) - 3) : 
                    mean.append(table.iloc[i, j+2])
                    period.append(j)
                    company.append(table.index[i])
            self.data_full.append(mean)
        self.data_full = pd.concat(
            [pd.DataFrame([company,period], index = ['company','period']).T,
             pd.DataFrame(self.data_full, index = df.columns[3:].tolist()).T]
            ,axis = 1)

    
    def creat_X_Y(self) : 
        data = pd.concat([self.data_full, self.distress], axis = 1).dropna(axis = 0)
        self.X = data.iloc[:,:-4].reset_index()
        self.Y = pd.DataFrame(list(np.where((data.iloc[: ,-2] < float(0)) & (data.iloc[:, -1]< float(0)) ,1,0)), 
                              columns = ['distress'])
        
        
    def creat_train_test(self, ratio) : 
        a = random.randint(1,1000)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X.iloc[:,3:], 
                                                                                self.Y, test_size = ratio,
                                                                                random_state = a)
        
    
    def clean_data(self, name1, name2, num,train , test) : 
        train.loc[:, name1] += num
        train.loc[:, name2] = np.log(train.loc[:, name2])
        
        ## drop 離群值 3 * std
        drop_index = set()
        for i in train.columns[:-9] : 
            mean = train.loc[:,i].mean()
            std = train.loc[:, i].std()
            index = train.loc[(train.loc[:,i] <= mean - std*3) | (train.loc[:,i] >= mean + std*3)].index
            drop_index.update(index.values)
        df = pd.concat([train, test], axis = 1).drop(drop_index, axis = 0)
        train = df.iloc[:,:-1]
        test = df.iloc[:,-1:]
        return train, test
        

