#!/usr/bin/env python
# coding: utf-8

# In[201]:


import numpy as np
import matplotlib.pyplot as plt
import random
import stackprinter
import seaborn as sns
stackprinter.set_excepthook(style='darkbg2')


# In[15]:


test = np.load('/Users/tommy84729/python/DL/HW1/test.npz')
train = np.load('/Users/tommy84729/python/DL/HW1/train.npz')


# In[16]:


x_train = train['image'].reshape(12000,784)
x_test = test['image'].reshape(5768,784)
x_train = x_train/255
x_test = x_test/255
y_train = np.zeros((len(x_train),10))
y_test = np.zeros((len(x_test),10))

for i in range(len(x_train)) : 
    y_train[i,int(train['label'][i])] = 1
for i in range(len(x_test)) : 
    y_test[i,int(test['label'][i])] = 1


# In[6]:


y_train.shape


# In[7]:


for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(train["image"][i])
    plt.title(int(train["label"][i]))
    plt.axis('off')
plt.show()


# ## DNN

# In[459]:


class dnn : 
    def __init__(self,x_train, x_test, y_train, y_test, para_init, dnn_arch) : 
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.layer = len(dnn_arch)
        self.dnn_arch = dnn_arch
        self.activation = 'relu & softmax'
        self.cost_function = 'cross_entropy'
        self.sample_size = x_train.shape[0]
        
        #initial parameter--------------------------------------------------
        

        if para_init == 'random' : 
            self.weight = [np.random.randn(dnn_arch[i-1],dnn_arch[i]) for i in range(1,len(dnn_arch))] *0.01
            self.bias = [np.random.randn(1,dnn_arch[i]) for i in range(1,len(dnn_arch))]*0.01
        elif para_init == 'zero' : 
            self.weight = [np.zeros(dnn_arch[i-1],dnn_arch[i]) for i in range(1,len(dnn_arch))]
            self.bias = [np.zeros(1,dnn_arch[i]) for i in range(1,len(dnn_arch))]
                           
    def softmax(self,x) : 
        shift = x - np.max(x)
        probs = np.exp(shift)/np.sum(np.exp(shift))                                                         
                           
        return probs
    def relu(self,x)  : 
        x = np.where(x > 0, x, x * 0.05) 
        return(x)
    
    def relu_backward(self,x) : 
        x[x >= 0] = 1
        x[x < 0]  = 1/10
        return x
    
    #backpropagation----------------------------------------------------------
    #forward--------------------------------
        
    def forward(self, x) : 
        self.layer_z = []
        self.layer_a = [x]
        for i in range(self.layer -1) : 
            x = np.dot(x,self.weight[i]) + self.bias[i]
            self.layer_z.append(x)

            x = self.relu(x)
            self.layer_a.append(x) ## activation fun

        self.output = np.apply_along_axis(self.softmax,1,x)
        
        return
    #Loss-------------------------------------
    def loss(self,y) : 
        self.error = -np.sum(np.multiply(np.log(self.output),y))/y.shape[1]
              
    #backward---------------------------------
    
    def backward(self, y,learning_rate ) : 
        back_input = (self.output - y)
        self.partial_loss = [back_input]
        x = back_input
        
        for i in reversed(range(self.layer - 2 )) : 
            b = self.relu_backward(self.layer_z[i])
            x = np.multiply(np.dot(x, self.weight[i+1].T), b)
            self.partial_loss.append(x)
            
        for i in range(self.layer - 1) : 
            self.weight[i] -= np.dot(self.layer_a[i].T,self.partial_loss[(-1-i)])*learning_rate/self.sample_size
            self.bias[i] -= self.partial_loss[(-1-i)].sum(0).reshape(1,-1)*learning_rate/self.sample_size
        
    #predict-----------------------------------
    def predict(self,x) : 
        for i in range(self.layer - 1) : 
            x = np.dot(x, self.weight[i]) + self.bias[i]
            x = self.relu(x)
            
        self.prediction = np.apply_along_axis(self.softmax, 1, x)
        return(np.argmax(self.prediction, 1))
            
            
            
      #model-----------------------------------
    
    def train(self, epoch = 10, batch = 400, learning_rate = 0.1, latent = [20,40]) : 
        self.cost = []
        self.train_error_rate = []
        self.test_error_rate = []
        self.latent_features = []
        for i in range(epoch) : 
            index = [j for j in range(1,self.sample_size)]
            random.shuffle(index)
            self.mini_loss = []
            if (i+1)%10 == 0 : 
                print('epoch' + str(i+1))
            for k in range(0, 30) : 
                x = self.x_train[index[k*400: (k+1)*400]]
                y = self.y_train[index[k*400: (k+1)*400]]
                
                self.forward(x)
                self.loss(y)
                self.mini_loss.append(self.error)
                
                self.backward(y, learning_rate = learning_rate)
            if self.dnn_arch[-2] == 2 : 
                if (i+1) == latent[0] or (i+1) == latent[1]  : 
                    k = self.x_test 
                    for q in range(self.layer-2) : 
                        k = np.dot(k, self.weight[q]) + self.bias[q]
                    self.latent_features.append(k)
                    
            train_y_hat = self.predict(self.x_train)
            test_y_hat = self.predict(self.x_test)
            self.cost.append(np.mean(self.mini_loss))
            self.train_error_rate.append(1 - (model.predict(x_train) == np.argmax(y_train,1)).sum()/self.sample_size)
            self.test_error_rate.append(1 - (model.predict(x_test) == np.argmax(y_test,1)).sum()/len(self.x_test))
         
        #laten
                   


# In[460]:


model = dnn(x_train, x_test,y_train, y_test, 'random', [784,256,10])
model.train(epoch = 200, batch = 400, learning_rate = 0.1,latent = [20,200])


# In[467]:


fig,ax= plt.subplots()
plt.plot(range(200),model.cost)
plt.title('training Loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
fig.savefig('/Users/tommy84729/Desktop/深度學習/HW_1/train_loss.png',dpi = 600)


# In[471]:


fig,ax= plt.subplots()
tr_error = plt.plot(range(200),model.train_error_rate,'b-',label = 'train_error_rate')
ts_error = plt.plot(range(200), model.test_error_rate, 'r-', label = 'test_error_rate')
plt.xlabel('epochs')
plt.ylabel('rate')
plt.title('Error Rate')
plt.legend()
#fig.savefig('/Users/tommy84729/Desktop/深度學習/HW_1/error_rate.png',dpi = 600)


# ## confusion matrix

# In[451]:


y_actu = pd.Series(test['label'].reshape(5768,).astype('int'), name='Actual')
y_pred = pd.Series(model.predict(x_test), name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_confusion


# In[452]:


model = dnn(x_train, x_test,y_train, y_test, 'random', [784,256,2,10])


# In[453]:


model.train(epoch = 200, batch = 400, learning_rate = 1,latent = [20,200])


# ## Latent Features

# In[454]:


latent_1 = pd.DataFrame(model.latent_features[0])
latent_1 = pd.concat([latent_1,pd.DataFrame(test['label'].reshape(5768,1))], axis = 1)
latent_1.columns = ['x', 'y', 'category']
latent_1['category'] = latent_1['category'].astype('int')
latent_1


# In[455]:


latent_2 = pd.DataFrame(model.latent_features[1])
latent_2 = pd.concat([latent_2,pd.DataFrame(test['label'].reshape(5768,1))], axis = 1)
latent_2.columns = ['x', 'y', 'category']
latent_2['category'] = latent_2['category'].astype(int)
latent_2


# In[456]:


latent_1_plot = pd.DataFrame()
for i in range(10) : 
    data_latent = latent_1[latent_1['category'] == i].reset_index()
    index = [i for i in range(len(latent_1[latent_1['category'] == i]))]
    latent_1_plot = pd.concat([latent_1_plot,data_latent.iloc[index[:150]]],axis = 0)
    
latent_2_plot = pd.DataFrame()
for i in range(10) : 
    data_latent = latent_2[latent_2['category'] == i].reset_index()
    index = [i for i in range(len(latent_2[latent_2['category'] == i]))]
    latent_2_plot = pd.concat([latent_2_plot,data_latent.iloc[index[:150]]],axis = 0)


# In[457]:


classification = [i for i in range(10)]
cmap = sns.color_palette("bright", n_colors=10, desat=.5)
ax = sns.scatterplot(x="x", y="y", hue = "category", 
                     data=latent_1_plot, legend = 'full',palette = cmap)
plt.legend( loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


# In[458]:


classification = [i for i in range(10)]
cmap = sns.color_palette("bright", n_colors=10, desat=.5)
ax = sns.scatterplot(x="x", y="y", hue = "category", 
                     data=latent_2_plot, legend = 'full',palette = cmap)
plt.legend( loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


# In[473]:


np.zeros([2,3])


# In[ ]:




