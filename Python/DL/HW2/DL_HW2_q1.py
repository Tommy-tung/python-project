#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import seaborn as sns
import stackprinter
import pygal
import pygal_maps_world.maps
from pygal_maps_world.i18n import COUNTRIES
import cairosvg
stackprinter.set_excepthook(style='darkbg2')


# In[3]:


data =pd.read_csv('/Users/tommy84729/Coding/DL/ＨＷ2/covid_19.csv') 
data = data.iloc[2:,:]


# In[4]:


data.index = data.iloc[:,0]
data = data.iloc[:,3:]
data = data.astype(float)


# In[5]:


data


# In[6]:


corr = data.T.corr()


# In[7]:


corr


# ## correlation figure

# In[8]:


plt.subplots(figsize=(20, 20)) # 設定畫面大小
sns.heatmap(corr, annot=True, vmax=1, square=True, cmap="Blues")


# In[ ]:


'''
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap = True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=False, linewidths=.5, cbar_kws={"shrink": .5})


# ## choose corr > 0.95 : 182

# In[17]:


country = []
for i in range(len(corr)) : 
    for j in range(len(corr)) : 
        if j != i : 
            if corr.iloc[i,j] > 0.95 : 
                if corr.index[j] not in country : 
                    country.append(corr.index[j])
                if corr.columns[i] not in country : 
                    country.append(corr.columns[i])


# In[18]:


len(country)


# ## 選定區間的大小：5

# In[21]:


def dataset(data, sequence) : 
    dataset = []
    label = []
    for i in range(len(data)) : 
        a = pd.DataFrame(df.iloc[i])
        for j in range(len(a)-sequence) : 
            x = a.iloc[j:j+sequence,0].values
            num = a.iloc[j+sequence,0]
            dataset.append(x)
            if x[4] < num : 
                label.append(1)
            else : 
                label.append(0)
    return(np.array(dataset), np.array(label))


# In[22]:


X, Y = dataset(df,5)


# In[23]:


X = np.array(X).reshape(-1,1,5)
X = X.astype(np.float32)


# ## train test split

# In[24]:


x_train = X[:int(13706 * 0.7)]
x_test = X[int(13706 * 0.7):]
y_train = Y[:int(13706 * 0.7)]
y_test = Y[int(13706 * 0.7):]


# In[25]:


x_train.shape


# In[26]:


batch_size = 32
feature_train = torch.from_numpy(x_train)
feature_test = torch.from_numpy(x_test)
target_train = torch.from_numpy(y_train).type(torch.LongTensor)
target_test = torch.from_numpy(y_test).type(torch.LongTensor)
train_set = torch.utils.data.TensorDataset(feature_train, target_train)
test_set = torch.utils.data.TensorDataset(feature_test, target_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, drop_last = False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True, drop_last = False)


# ## Model 

# In[27]:


class basemodel(nn.Module) : 
    def __init__(self, input_dim, hidden, output_dim, layer, model_type) : 
        super(basemodel, self).__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.layer = layer
        if model_type == 'Rnn' : 
            self.model = nn.RNN(input_size=self.input_dim, hidden_size=self.hidden,
                        num_layers=self.layer, dropout=0.0,
                         nonlinearity="tanh", batch_first = True)
        if model_type == 'Lstm' : 
            self.model = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden,
                        num_layers=self.layer, dropout=0.0,
                         batch_first=True)
        self.fc = nn.Linear(self.hidden, self.output_dim)
        print(self.model)
            


# In[28]:


class rnnmodel(basemodel) : 
    def __init__(self,input_dim, hidden, output_dim, layer, model_type) : 
        super(rnnmodel, self).__init__(input_dim, hidden, output_dim, layer, model_type)
        
    def forward(self, x) : 
        h0 = Variable(torch.randn(self.layer , x.size(0), self.hidden))
        rnn_output, hn = self.model(x, h0)
        out = rnn_output[:, -1 , :]
        out = self.fc(out)
        out = F.softmax(out, dim = 1)
        hn = hn.view(x.size(0),self.hidden)
        
        return out


# In[29]:


class lstm(basemodel) : 
    
    def __init__(self,input_dim, hidden, output_dim, layer, model_type) : 
        super(lstm, self).__init__(input_dim, hidden, output_dim, layer, model_type)
        
    def forward(self, x) : 
        h0 = torch.randn(self.layer , x.size(0), self.hidden)
        c0 = torch.randn(self.layer , x.size(0), self.hidden)
        rnn_output, (hn, cn) = self.model(x, (h0, c0))
        out = rnn_output[ :, -1, : ]
        out = self.fc(out)
        out = F.softmax(out, dim = 1)
        hn = hn.view(x.size(0),self.hidden)
        return out
        
    def predict(self, x):
        outputs = self(x)
        _, predicted = torch.max(outputs, 1)
        return predicted


# In[30]:


def train(model,train_loader,test_loader, epoch) : 
    train_acc = []
    test_acc = []
    criterion = nn.NLLLoss()
    for i in range(epoch) : 
        correct_train = 0
        correct_test = 0
        total_train = 0
        total_test = 0
        label_train = []
        label_test = []
        print(f'epoch: {i+1}/{epoch}', end = '\r')

        for x_train ,label in (train_loader) : 

            train_pred = model(x_train)
            optimizer.zero_grad()
            loss = criterion(train_pred, label)
            # Backward pass
            loss.backward()
            optimizer.step()
            predict = torch.max(train_pred.data,1)[1]
            label_train.extend(label.cpu().numpy())
            total_train += label.size(0)
            correct_train += (predict == label).sum().item()
        train_acc.append(correct_train/total_train)
        model.eval()
        for x_test, label in (test_loader) : 
            test_pred = model(x_test)
            test_predict = torch.max(test_pred.data,1)[1]
            label_test.extend(label.cpu().numpy())
            total_test += label.size(0)
            correct_test += (test_predict == label).sum().item()
        test_acc.append(correct_test/total_test)
    
    return train_acc, test_acc

    


# In[31]:


model_rnn = rnnmodel(5,32,2,1,'Rnn').float()
optimizer = torch.optim.Adam(model_rnn.parameters(), lr = 0.001)


# In[32]:


rnn_train_acc, rnn_test_acc = train(model_rnn,train_loader,test_loader,100)


# In[33]:


plt.plot(rnn_train_acc)


# In[34]:


plt.plot(rnn_test_acc)


# ## change different time interval : 7

# In[37]:


X_7, Y_7 = dataset(df,7)
X_7 = np.array(X_7).reshape(-1,1,7)
X_7 = X_7.astype(np.float32)


# In[38]:


len(X_7)


# In[39]:


index = [i for i in range(len(X_7))]
np.random.shuffle(index)
x_train_7 = X_7[index[ : int(len(X_7) * 0.7)]]
x_test_7 = X_7[index[int(len(X_7) * 0.7) : ]]
y_train_7 = Y_7[  index[:int(len(X_7) * 0.7)]]
y_test_7 = Y_7 [index[int(len(X_7) * 0.7) : ]]


# In[40]:


feature_train = torch.from_numpy(x_train_7)
feature_test = torch.from_numpy(x_test_7)
target_train = torch.from_numpy(y_train_7).type(torch.LongTensor)
target_test = torch.from_numpy(y_test_7).type(torch.LongTensor)


# In[41]:


train_set = torch.utils.data.TensorDataset(feature_train, target_train)
test_set = torch.utils.data.TensorDataset(feature_test, target_test)


# In[42]:


train_loader_7 = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = True, drop_last = False)
test_loader_7 = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = True, drop_last = False)


# In[43]:


model_lstm = lstm(7,32,2,1,'Lstm').float()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr = 0.001)


# In[44]:


lstm_train_acc, lstm_test_acc = train(model_lstm,train_loader_7,test_loader_7,100)


# In[45]:


plt.plot(lstm_train_acc)


# In[46]:


plt.plot(lstm_test_acc)


# ## predict with probability and Map

# In[47]:


pred_data = df.iloc[:, -7:]
pred_data = np.array(pred_data).reshape(-1,1,7)
#pred_data = torch.from_numpy(pred_data)
#pred_data = torch.utils.data.TensorDataset(pred_data)
pred_data = Variable(torch.LongTensor(pred_data).float())
#pred_data= torch.utils.data.DataLoader(pred_data, batch_size = 1 , shuffle = False, drop_last = False)
#dataiter = iter(pred_data) 
#data = dataiter.next()


# In[48]:


pred_data.shape


# In[50]:


model_lstm.eval()
output = model_lstm(pred_data)


# In[51]:


country_set = pd.DataFrame(COUNTRIES.items())
prob = pd.DataFrame(output.detach().numpy()[:,:])
country = pd.DataFrame(df.index.values)
prob = pd.concat([country, prob], axis = 1)
prob.columns = ['country', 'desc', 'asc']
country_set.columns = ['code', 'country']
prob = pd.merge(prob, country_set, left_on = 'country', right_on = 'country', how = 'left')


# In[52]:


ascending = []
descending = []
for i in range(len(prob)) : 
    if prob.iloc[i,1] > 0.5 : 
        descending.append(i)
    else : 
        ascending.append(i)


# In[53]:


ascending = prob.iloc[ascending, : ]
descending = prob.iloc[descending, : ]


# In[54]:


ascending


# In[55]:


dict_asc = {}
keys = [i for i in ascending.iloc[:,3]]
values = [i for i in ascending.iloc[:,2]]
for i ,name in enumerate(keys):
        dict_asc[name] = values[i]

dict_desc = {}
keys = [i for i in descending.iloc[:,3]]
values = [i for i in descending.iloc[:,1]]
for i ,name in enumerate(keys):
        dict_desc[name] = values[i]


# In[65]:


wm = pygal_maps_world.maps.World()
wm.add('ascending', dict_asc)
wm.add('descending', dict_desc)


# In[57]:


country_set = pd.DataFrame(COUNTRIES.items())
country_set

