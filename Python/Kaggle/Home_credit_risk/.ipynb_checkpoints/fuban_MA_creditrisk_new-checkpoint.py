#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd 
import numpy as np
import os 
from sklearn.impute import SimpleImputer


# In[136]:


path = '/Users/tommy84729/Coding/Fubon/home-credit-default-risk'
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


# In[3]:


### train/test : 客戶申請表
### bureau/bureau_balance 由其他金融机构提供给征信中心的客户信用记录历史(月数据)
### cash_balance 客户在Home Credit数据库中POS(point of sales)和现金贷款历史(月数据)
### credit_card 客户在Home Credit数据库中信用卡的snapshot历史(月数据) 包含了客户消费次数, 消费金额等情况
### previous(application) 客户先前的申请记录，包含了客户所有历史申请记录(申请信息, 申请结果等)
### installments_payment 客户先前信用卡的还款记录，包含了客户的还款情况(还款日期, 是否逾期, 还款金额, 是否欠款等)


# In[43]:


filelist = os.listdir(path)
test = pd.read_csv(path+'/'+filelist[0])
cash_balance = pd.read_csv(path+'/'+filelist[2])
credit_card = pd.read_csv(path+'/'+filelist[3])
installment_payment = pd.read_csv(path+'/'+filelist[4])
train = pd.read_csv(path+'/'+filelist[5])
bureau = pd.read_csv(path+'/'+filelist[6])
previous = pd.read_csv(path+'/'+filelist[7])
bureau_balance = pd.read_csv(path+'/'+filelist[8])


# In[135]:


class data_preprocessing : 
    
    def __init__(self, train, test, cash_balance, credit_card, installment_payment, bureau, bureau_balance, previous, numerics) : 
        
        self.train = train
        self.test = test
        self.cash_balance = cash_balance
        self.credit_card = credit_card
        self.installment_payment = installment_payment
        self.bureau = bureau
        self.bureau_balance = bureau_balance
        self.previous = previous
        self.numerics = numerics
        
    def one_hot_encoder(self, df, nan_as_category = False):
        original_columns = list(df.columns) # col names as string in a list 
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object'] #categorical col names
        df = pd.get_dummies(df, columns = categorical_columns, dummy_na = nan_as_category) #creating dummies
        return df
    
    ## 使用出現次數最高來對類別變數補植
    def class_na(self, df) :
        for i in df.select_dtypes(include = object).columns : 
            if df[i].isna().sum() == 0 : 
                continue
            else :
                df[i] = df[i].fillna(df[i].value_counts().index[0])
        return df

    def application_preprocessing(self):
        
        self.train = self.class_na(self.train)
        self.test = self.class_na(self.test)
        ## 數值變數：平均 or 0 （OWN_CAR_AGE）
        self.train.loc[:,'OWN_CAR_AGE']  = self.train.loc[:,'OWN_CAR_AGE'].fillna(0)
        self.test.loc[:,'OWN_CAR_AGE']  = self.test.loc[:,'OWN_CAR_AGE'].fillna(0)
        imr = SimpleImputer(missing_values = np.nan, 
                            strategy = 'mean')
        self.train[self.train.select_dtypes(include=numerics).columns] = imr.fit_transform(self.train.select_dtypes(include=numerics).values)
        self.test[self.test.select_dtypes(include=numerics).columns] = imr.fit_transform(self.test.select_dtypes(include=numerics).values)
        
        ## 合併train & test 進行feature engineering
        df = self.train.append(self.test).reset_index().drop('index', axis = 1)

        df['APP_YEAR'] = df['DAYS_BIRTH'] / (- 365)
        df['APP_DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        df['APP_INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['APP_ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['APP_PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df["APP_CREDIT_GOODS_PRICE_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
        df["APP_GOODS_CREDIT_DIFF"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
        df['APPS_INCOME_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL']/df['DAYS_BIRTH']
        df['APP_CNT_ADULT'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
        df['APPS_CNT_FAM_INCOME_RATIO'] = df['AMT_INCOME_TOTAL']/df['CNT_FAM_MEMBERS']

        self.application = df.drop('DAYS_BIRTH', axis = 1)
        
        print('application_preprocessing finished')
        print('====================================')
        
    def bureau_bureau_bal_preprocessing(self) : 
        
        ## bureau
        self.bureau = self.bureau.fillna(0)
        
        self.bureau['BUREAU_CREDIT_DEBT_DIFF'] = self.bureau['AMT_CREDIT_SUM_DEBT'] - self.bureau['AMT_CREDIT_SUM']

        bureau_agg_dict = {
          'SK_ID_BUREAU':['count'],
          'DAYS_CREDIT':['min', 'max', 'mean'],
          'CREDIT_DAY_OVERDUE':['min', 'max', 'mean'],
          'DAYS_CREDIT_ENDDATE':['min', 'max', 'mean'],
          'DAYS_ENDDATE_FACT':['min', 'max', 'mean'],
          'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
          'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
          'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
          'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
          'AMT_ANNUITY': ['max', 'mean', 'sum'],

          # new feature

          'BUREAU_CREDIT_DEBT_DIFF':['min', 'max', 'mean'],
          }

        bureau_agg = self.bureau.groupby(['SK_ID_CURR']).agg(bureau_agg_dict)
        bureau_agg.columns = ['BUREAU_'+('_').join(column).upper() for column in bureau_agg.columns.ravel()]
        self.bureau_agg = bureau_agg.reset_index()

        active = self.bureau[self.bureau['CREDIT_ACTIVE'] == 'Active']
        bureau_active_agg = active.groupby(['SK_ID_CURR']).agg(bureau_agg_dict)
        bureau_active_agg.columns = ['BUREAU_ACT_'+('_').join(column).upper() for column in bureau_active_agg.columns.ravel()]
        self.bureau_active_agg = bureau_active_agg.reset_index()
        
        
        ## bureau_balance
        bureau_bal = self.bureau_balance.merge(self.bureau[['SK_ID_CURR', 'SK_ID_BUREAU']], 
                                               on='SK_ID_BUREAU', 
                                               how='left')
        ## New Features
        bureau_bal['BUREAU_BAL_IS_DPD'] = bureau_bal['STATUS'].apply(lambda x: 1 if x in['1','2','3','4','5']  else 0)
        bureau_bal['BUREAU_BAL_IS_DPD_OVER120'] = bureau_bal['STATUS'].apply(lambda x: 1 if x =='5'  else 0)

        ##Aggregate
        bureau_bal_agg_dict = {
            'SK_ID_CURR':['count'],
            'MONTHS_BALANCE':['min', 'max', 'mean'],
            'BUREAU_BAL_IS_DPD':['mean', 'sum'],
            'BUREAU_BAL_IS_DPD_OVER120':['mean', 'sum']
         }
        
        bureau_bal_agg = bureau_bal.groupby('SK_ID_CURR').agg(bureau_bal_agg_dict)
        bureau_bal_agg.columns = [ 'BUREAU_BAL_'+('_').join(column).upper() for column in bureau_bal_agg.columns.ravel() ]
        self.bureau_bal_agg = bureau_bal_agg.reset_index()
            
        print(' bureau_bureau_bal_preprocessing finished')
        print('====================================')
        
    def previous_preprocessing(self) : 
        
        ##取代值365243
        self.previous.replace(365243,
                              np.nan,
                              inplace = True)
        self.previous = self.class_na(self.previous)
        self.previous = self.previous.fillna(0)
        previous = self.previous
        previous['PREV_CREDIT_DIFF'] = previous['AMT_APPLICATION'] - previous['AMT_CREDIT']
        previous['PREV_GOODS_DIFF'] = previous['AMT_APPLICATION'] - previous['AMT_GOODS_PRICE']
        previous['PREV_CREDIT_APPL_RATIO'] = previous['AMT_CREDIT']/previous['AMT_APPLICATION']
        previous['PREV_GOODS_APPL_RATIO'] = previous['AMT_GOODS_PRICE']/previous['AMT_APPLICATION'] ## 實際貸款金額/申請貸款金額

        ## 減少one hot encoder變數量 - NAME_GOODS_CATEGORY
        a = ['Auto Accessories', 'Jewelry', 'Homewares', 
             'Medical Supplies', 'Vehicles', 'Sport and Leisure','Gardening', 
             'Other', 'Office Appliances', 'Tourism', 'Medicine', 'Direct Sales', 
             'Fitness', 'Additional Service','Education', 'Weapon', 'Insurance', 
             'House Construction', 'Animals']  
        previous["NAME_GOODS_CATEGORY"] = previous["NAME_GOODS_CATEGORY"].replace(a, 'Other_GOODS')
        
        ## NAME_CASH_LOAN_PURPOSE
        a = ['Buying a used car','Building a house or an annex','Everyday expenses','Medicine',
             'Payments on other loans','Education','Journey', 'Purchase of electronic equipment',
             'Buying a new car','Wedding / gift / holiday','Buying a home','Car repairs','Furniture',
             'Buying a holiday home / land', 'Business development','Gasification / water supply',
             'Buying a garage','Hobby','Money for a third person','Refusal to name the goal',
             'Urgent needs','Other']
        previous['NAME_CASH_LOAN_PURPOSE']= previous['NAME_CASH_LOAN_PURPOSE'].replace(a,'Others')
        self.previous = previous
        
        agg_dict = {
          # 原始變數
          'SK_ID_CURR':['count'],
          'AMT_CREDIT':['mean', 'max', 'sum'],
          'AMT_ANNUITY':['mean', 'max', 'sum'], 
          'AMT_APPLICATION':['mean', 'max', 'sum'],
          'AMT_DOWN_PAYMENT':['mean', 'max', 'sum'],
          'AMT_GOODS_PRICE':['mean', 'max', 'sum'],
          'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
          'DAYS_DECISION': ['min', 'max', 'mean'],
          'CNT_PAYMENT': ['mean', 'sum'],

          # NEW feature
          'PREV_CREDIT_DIFF':['mean', 'max', 'sum'], 
          'PREV_CREDIT_APPL_RATIO':['mean', 'max'],
          'PREV_GOODS_DIFF':['mean', 'max', 'sum'],
          'PREV_GOODS_APPL_RATIO':['mean', 'max'],
        }

        prev_agg = previous.groupby('SK_ID_CURR').agg(agg_dict)
        prev_agg.columns = ["PREV_"+ "_".join(x).upper() for x in prev_agg.columns.ravel()]
        ## refused rate
        previous_contract_status = previous.groupby('SK_ID_CURR').agg({'NAME_CONTRACT_STATUS' : ['count']})
        previous_contract_status_refused = previous[previous.NAME_CONTRACT_STATUS == 'Refused'].groupby('SK_ID_CURR').agg({'NAME_CONTRACT_STATUS' : ['count']})
        previous_contract_status = previous_contract_status.join(previous_contract_status_refused,lsuffix='_left', rsuffix='_right')
        previous_contract_status = previous_contract_status.fillna(0)

        self.prev_agg = prev_agg.join(previous_contract_status.NAME_CONTRACT_STATUS_right/previous_contract_status.NAME_CONTRACT_STATUS_left)
        self.prev_agg = self.prev_agg.reset_index()
        
        print('previous_preprocessing finished')
        print('====================================')
        
    def cash_balance_preprocessing(self) : 
        
        self.cash_balance = self.cash_balance.fillna(0)
        
        cash_bal_agg = {
        'SK_ID_CURR' : ['size'],
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean', 'var'],
        'SK_DPD_DEF': ['max', 'mean', 'var']
        }
        cash_agg = self.cash_balance.groupby('SK_ID_CURR').agg(cash_bal_agg)
        cash_agg.columns = ["PREV_"+ "_".join(x).upper() for x in cash_agg.columns.ravel()]
        self.cash_agg = cash_agg.reset_index()
        
        print('cash_balance_preprocessing finished')
        print('====================================')
    
    def installment_payment_preprocessing(self) : 
        
        self.installment_payment = self.installment_payment.fillna(0)
        installment_payment = self.installment_payment
        installment_payment['PAYMENT_PERC'] = installment_payment['AMT_PAYMENT'] / installment_payment['AMT_INSTALMENT']
        installment_payment['PAYMENT_DIFF'] = installment_payment['AMT_INSTALMENT'] - installment_payment['AMT_PAYMENT']
        installment_payment['DPD'] = installment_payment['DAYS_ENTRY_PAYMENT'] - installment_payment['DAYS_INSTALMENT']
        installment_payment['DPD'] = installment_payment['DPD'].apply(lambda x: x if x > 0 else 0)
        
        self.installment_payment = installment_payment
        
        ins_pay_agg = {
            'SK_ID_CURR': ['size'],
            'DPD': ['max', 'mean', 'sum'],
            'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
            'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
            'AMT_INSTALMENT': ['max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
        }
        
        ins_agg = installment_payment.groupby('SK_ID_CURR').agg(ins_pay_agg)
        ins_agg.columns = pd.Index(['INSTAL_' + col[0] + "_" + col[1].upper() for col in ins_agg.columns.tolist()])
        self.ins_agg = ins_agg.reset_index()
        
        print('installment_payment_preprocessing finished')
        print('====================================')
    
    def credit_card_preprocessing(self) : 
        
        self.credit_card = self.credit_card.fillna(0)
        
        credi_card = self.credit_card
        credit_card['BALANCE_LIMIT_RATIO'] = credit_card['AMT_BALANCE']/credit_card['AMT_CREDIT_LIMIT_ACTUAL']
        credit_card['DRAWING_LIMIT_RATIO'] = credit_card['AMT_DRAWINGS_CURRENT'] / credit_card['AMT_CREDIT_LIMIT_ACTUAL']

        credit_card['CARD_IS_DPD'] = credit_card['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
        #credit_card['CARD_IS_DPD_UNDER_90'] = credit_card['SK_DPD'].apply(lambda x:1 if (x > 0) & (x <90) else 0 )
        #credit_card['CARD_IS_DPD_OVER_90'] = credit_card['SK_DPD'].apply(lambda x:1 if x >= 90 else 0)
        self.credit_card = credit_card
        
        # aggregation

        credit_card_agg_dict = {
            'SK_ID_CURR':['count'],
            'MONTHS_BALANCE':['sum', 'max', 'mean'],
            'AMT_BALANCE':['max'],
            'AMT_CREDIT_LIMIT_ACTUAL':['max'],
            'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
            'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
            'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['max','sum'],
            'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
            'CNT_DRAWINGS_ATM_CURRENT': ['max','sum'],
            'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
            'CNT_DRAWINGS_POS_CURRENT': ['mean'],
            'SK_DPD': ['mean', 'max', 'sum'],
            #  new feature
            'BALANCE_LIMIT_RATIO':['min','max'],
            'DRAWING_LIMIT_RATIO':['min', 'max'],
            'CARD_IS_DPD':['mean', 'sum'],
            #'CARD_IS_DPD_UNDER_90':['mean', 'sum'],
            #'CARD_IS_DPD_OVER_90':['mean', 'sum']    
        }
        credit_card_agg = credit_card.groupby('SK_ID_CURR').agg(credit_card_agg_dict)
        credit_card_agg.columns = ['CARD_'+('_').join(column).upper() for column in credit_card_agg.columns.ravel()]

        self.credit_card_agg = credit_card_agg.reset_index()
        
        print('credit_card_preprocessing finished')
        print('====================================')
        
    def create_train_test(self) : 
        
        data = self.one_hot_encoder(self.application)
        data = data.join(self.bureau_agg, how='left', on='SK_ID_CURR',rsuffix='_right')
        data = data.join(self.bureau_active_agg, how='left', on='SK_ID_CURR', rsuffix='_right')
        data = data.join(self.prev_agg, how='left', on='SK_ID_CURR', rsuffix='_right')
        data = data.join(self.cash_agg, how='left', on='SK_ID_CURR', rsuffix='_right')
        data = data.join(self.ins_agg, how='left', on='SK_ID_CURR', rsuffix='_right')
        data = data.join(self.credit_card_agg, how='left', on='SK_ID_CURR', rsuffix='_right')
        
        self.answer_id = data[data.TARGET.isna() == True].SK_ID_CURR.astype('int32')
        self.answer_id = pd.DataFrame(self.answer_id.reset_index(drop = True))
        data = data.iloc[:, 1:].drop('SK_ID_CURR', axis = 1)
        data = data.replace([np.inf,-np.inf],np.nan)
        train = data[data.TARGET.isna() == False].fillna(0)
        self.x_test = data[data.TARGET.isna() == True].drop('TARGET', axis = 1).fillna(0)
        self.y_train = train.TARGET
        self.x_train = train.drop('TARGET', axis = 1)
        
        print('train test split finished')
    

