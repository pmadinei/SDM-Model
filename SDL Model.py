#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,timedelta
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[3]:


def makeRandom(y_test):
    rands = []
    for counter in range(len(y_test)):
        rands.append(np.random.choice(list(range(1,300))))
    return rands

def reportRegressor(model,X_cross,y_cross,X_test,y_test):
    validationSetMSE = metrics.mean_squared_error(y_cross,model.predict(X_cross))
    validationSetR2 = metrics.r2_score(y_cross,model.predict(X_cross))
    testSetMSE = metrics.mean_squared_error(y_test,model.predict(X_test))
    testSetR2 = metrics.r2_score(y_test,model.predict(X_test))
    random_predicts = makeRandom(y_test)
    randomMSE = metrics.mean_squared_error(y_test,random_predicts)
    randomR2 = metrics.r2_score(y_test,random_predicts)
    print('Validation-set:\n\tMean Squared Error: ' , validationSetMSE , '\n\tR2 Score: ' , validationSetR2)
    print('\nTest-set:\n\tMean Squared Error: ' , testSetMSE , '\n\tR2 Score: ' , testSetR2)
    print('\nRandom Predicts on Test-set:\n\tMean Squared Error: ' , randomMSE , '\n\tR2 Score: ' , randomR2)


# In[76]:


# class predict_latency_model():
#     def __init__(self):
        
latencyDF = pd.read_csv('data.csv')
ss = StandardScaler()
temp = list(latencyDF.columns)
temp.remove('Latency')
ss = StandardScaler()
latencyDF[temp] = ss.fit_transform(latencyDF.drop('Latency',axis=1))


# In[582]:


# X_train = pd.DataFrame(columns=latencyDF.columns)
# for i in range(len(latencyDF)):
#     if ((i%7 != 6) and (i%7 != 5)):
#         X_train = X_train.append(latencyDF.iloc[i],ignore_index = False)


# In[583]:


# X_train.head(16)


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(latencyDF.drop('Latency',axis=1), latencyDF['Latency'], test_size=0.2, random_state=None, shuffle = True)
X_others, X_cross, y_others, y_cross = train_test_split(X_train, y_train, test_size=0.2, random_state=None, shuffle = True)


# # Decision Tree

# In[78]:


from sklearn.tree import DecisionTreeRegressor
validation_acc = {}
test_acc = {}
for i in range(1,20):
    decisionTree = DecisionTreeRegressor(max_depth=i)
    decisionTree.fit(X_train,y_train)
    validation_acc[i] = metrics.mean_squared_error(y_cross,decisionTree.predict(X_cross))
    test_acc[i] = metrics.mean_squared_error(y_test,decisionTree.predict(X_test))

plt.figure(figsize=(15,6))
plt.title('Decision Tree')
plt.xlabel('Max Depth')
plt.ylabel('Mean Squared Error')
plt.plot(list(validation_acc.keys()),list(validation_acc.values()),label = "Validation-set")
plt.plot(list(test_acc.keys()),list(test_acc.values()),label = "Test-set")
plt.tight_layout()
plt.legend()
plt.show()


# In[133]:


X_train, X_test, y_train, y_test = train_test_split(latencyDF.drop('Latency',axis=1), latencyDF['Latency'], test_size=0.2, random_state=None, shuffle = True)
X_others, X_cross, y_others, y_cross = train_test_split(X_train, y_train, test_size=0.2, random_state=None, shuffle = True)
X_train, X_test, y_train, y_test = train_test_split(latencyDF.drop('Latency',axis=1), latencyDF['Latency'], test_size=0.2, random_state=None, shuffle = True)
X_others, X_cross, y_others, y_cross = train_test_split(X_train, y_train, test_size=0.2, random_state=None, shuffle = True)
decisionTree = DecisionTreeRegressor(max_depth=10)
decisionTree.fit(X_train,y_train)
reportRegressor(decisionTree,X_cross,y_cross,X_test,y_test)


# ## Logistic Regression

# In[134]:


from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train,y_train)
reportRegressor(linear_regressor,X_cross,y_cross,X_test,y_test)


# # SVM
# 

# In[135]:


from sklearn.svm import SVR
# param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'] , 'epsilon' : [0.1,0.2,0.5,0.001,0.0001]} 
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=3)
# grid.fit(X_train,y_train)
# grid.best_params_
svr = SVR(C = 100 , epsilon=0.2,gamma=1,kernel='rbf')
svr.fit(X_train,y_train)
reportRegressor(svr,X_cross,y_cross,X_test,y_test)


# # Random Forrest

# In[136]:


from sklearn.ensemble import RandomForestRegressor
# param_grid = {'n_estimators': [10, 50, 100, 500, 1000], 'max_depth': [5,10,15,20,30]} 
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(RandomForestRegressor(),param_grid,refit=True,verbose=3)
# grid.fit(X_train,y_train)


# In[137]:


# grid.best_params_


# In[138]:


validation_acc = {}
test_acc = {}
for i in range(1,30):
    rfc = RandomForestRegressor(n_estimators=50,max_depth=i)
    rfc.fit(X_train,y_train)
    validation_acc[i] = metrics.mean_squared_error(y_cross,rfc.predict(X_cross))
    test_acc[i] = metrics.mean_squared_error(y_test,rfc.predict(X_test))

plt.figure(figsize=(15,6))
plt.title('Max Depth effect on Random Forests')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.plot(list(validation_acc.keys()),list(validation_acc.values()),label = "Validation-set")
plt.plot(list(test_acc.keys()),list(test_acc.values()),label = "Test-set")
plt.tight_layout()
plt.legend()
plt.show()


# In[139]:


validation_acc = {}
test_acc = {}
for i in range(1,80):
    rfc = RandomForestRegressor(n_estimators=i,max_depth=10)
    rfc.fit(X_train,y_train)
    validation_acc[i] = metrics.mean_squared_error(y_cross,rfc.predict(X_cross))
    test_acc[i] = metrics.mean_squared_error(y_test,rfc.predict(X_test))

plt.figure(figsize=(15,6))
plt.title('Max Depth effect on Random Forests')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.plot(list(validation_acc.keys()),list(validation_acc.values()),label = "Validation-set")
plt.plot(list(test_acc.keys()),list(test_acc.values()),label = "Test-set")
plt.tight_layout()
plt.legend()
plt.show()


# In[17]:


from mpl_toolkits import mplot3d

validation_acc = {}
test_acc = {}
for i in range(1,100):
    for j in range(1,30):
        rfc = RandomForestRegressor(n_estimators=i,max_depth=j)
        rfc.fit(X_train,y_train)
        validation_acc[(i,j)] = metrics.mean_squared_error(y_cross,rfc.predict(X_cross))
        test_acc[(i,j)] = metrics.mean_squared_error(y_test,rfc.predict(X_test))


# In[26]:


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111, projection='3d')

X = []
Y = []
for i in list(validation_acc.keys()):
    X.append(i[0])
    Y.append(i[1])
ax.scatter(X,Y,list(validation_acc.values()),c='skyblue', s=40,label='Validation Set')

X = []
Y = []
for i in list(test_acc.keys()):
    X.append(i[0])
    Y.append(i[1])
ax.scatter(X,Y,list(test_acc.values()),c='red', s=40,label='Test Set')

ax.set_xlabel('N_Estimators')
ax.set_ylabel('Max_Depth')
ax.set_zlabel('Mean Squared Error');
plt.title('n_estimators and Max_Depth effect on Random Forests')
plt.tight_layout()
plt.legend()
plt.show()


# In[140]:


rfr = RandomForestRegressor(n_estimators=20,max_depth=9)
rfr.fit(X_train,y_train)
reportRegressor(rfr,X_cross,y_cross,X_test,y_test)


# # KNN

# In[141]:


from sklearn.neighbors import KNeighborsRegressor
validation_acc = {}
test_acc = {}
for i in range(1,30):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train,y_train)
    validation_acc[i] = metrics.mean_squared_error(y_cross,knn.predict(X_cross))
    test_acc[i] = metrics.mean_squared_error(y_test,knn.predict(X_test))

plt.figure(figsize=(15,6))
plt.title('n_neighbors effect on KNNs')
plt.xlabel('n_neighbors')
plt.ylabel('Mean Squared Error')
plt.plot(list(validation_acc.keys()),list(validation_acc.values()),label = "Validation-set")
plt.plot(list(test_acc.keys()),list(test_acc.values()),label = "Test-set")
plt.tight_layout()
plt.legend()
plt.show()


# In[142]:


knn = KNeighborsRegressor(n_neighbors=6)
knn.fit(X_train,y_train)
reportRegressor(knn,X_cross,y_cross,X_test,y_test)


# # Voting

# In[143]:


from sklearn.ensemble import VotingRegressor
vc = VotingRegressor(estimators=[('rf',rfr) , ('kn',knn) , ('sv',svr), ('dt',decisionTree)])
vc.fit(X_train,y_train)
reportRegressor(vc,X_cross,y_cross,X_test,y_test)


# In[272]:


a = []
b = []
for i in (decisionTree.predict(X_test)):
    if i>150:
        a.append('Memory')
    else:
        a.append('Amnesia')
for i in y_test:
    if i>150:
        b.append('Memory')
    else:
        b.append('Amnesia')
print(metrics.classification_report(b,a))


# In[504]:


acc = []
preds = decisionTree.predict(X_test)
for i in range(len(preds)):
    if (abs(preds[i] - list(y_test)[i]) < 100):
        acc.append('True')
    else:
        acc.append('False')
acc.count('True')/len(acc)


# # Neural Networks

# In[280]:


import random
import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import time
from datetime import timedelta
from datetime import datetime
from sklearn.model_selection import train_test_split


# In[281]:


class Model(nn.Module):
    def __init__(self, class_num, act=F.relu):

        super(Model, self).__init__()

        self.layer1 = nn.Linear(1 * 20, 4000)
        self.act1 = act

        self.layer2 = nn.Linear(4000, 2000)
        self.act2 = act


        self.layer3 = nn.Linear(2000, 1000)
        self.act3 = act

        self.layer4 = nn.Linear(1000, 500)
        self.act4 = act

        self.layer5 = nn.Linear(500, 250)
        self.act5 = act

        self.layer6 = nn.Linear(250, 1)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        #Make it one-dimentional

        x = self.layer1(x)
        x = self.act1(x)

        x = self.layer2(x)
        x = self.act2(x)

        x = self.layer3(x)
        x = self.act3(x)

        x = self.layer4(x)
        x = self.act4(x)

        x = self.layer5(x)
        x = self.act5(x)

        x = self.layer6(x)
        return x

    
    def start_weights(self, constant):
  
        nn.init.constant_(self.layer1.weight, constant)
        self.layer1.bias.data.fill_(0)

        nn.init.constant_(self.layer2.weight, constant)
        self.layer2.bias.data.fill_(0)

        nn.init.constant_(self.layer3.weight, constant)
        self.layer3.bias.data.fill_(0)

        nn.init.constant_(self.layer4.weight, constant)
        self.layer4.bias.data.fill_(0)

        nn.init.constant_(self.layer5.weight, constant)
        self.layer5.bias.data.fill_(0)

        nn.init.constant_(self.layer6.weight, constant)
        self.layer6.bias.data.fill_(0)


# In[ ]:


def fit(model, train_loader, device, criterion, optimizer, num_epochs=10):

    total_time = 0.

    # For the use of the function "plot_loss_changes" 
    loss_epoch = []

    for epoch in range(num_epochs):
        train_loss = 0.
        d1 = datetime.now()
        for images, labels in train_loader:
            #Go to GPU
            images = images.to(device)
            labels = labels.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        # images = images.view(1, -1)
        outputs = model(images.float())

        # Calculate Loss: softmax --> cross entropy loss
        # loss here is a tensor
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters with back propagation
        loss.backward()

        # Updating parameters
        optimizer.step()
        train_loss += loss.item()

      # statistics
      average_loss = train_loss / len(train_loader)

      loss_epoch.append(tuple([average_loss,epoch + 1]))

      d2 = datetime.now()
      delta = d2 - d1
      seconds = float(delta.total_seconds())
      total_time += seconds
      print('epoch %d, train_loss: %.3f, time elapsed: %s seconds' % (epoch + 1, average_loss, seconds))
  print('total training time: %.3f minutes' % (total_time / 60))
  return loss_epoch


# In[283]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer


# In[491]:


model = Sequential()
model.add(Dense(20, activation='relu', input_dim=20 ))
model.add(Dropout(0.1))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(1, activation='relu'))
model.summary()

# lr_schedule = keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-2,
#     decay_steps=10000,
#     decay_rate=0.9)
# opt = keras.optimizers.SGD(learning_rate=lr_schedule)
# model.compile(loss='mean_squared_error', optimizer=opt)

# Compiling the model using categorical_crossentropy loss, and rmsprop optimizer.
model.compile(loss='mean_squared_error',
              optimizer='Adam',
              metrics=['mse'])


# In[477]:


keras._estimator_type = 'regressor'


# In[479]:


hist = model.fit(X_train, y_train,batch_size=128,epochs=110,validation_data=(X_cross, y_cross),verbose=2,validation_split=0.1,workers=10)


# In[480]:


reportRegressor(model,X_cross,y_cross,X_test,y_test)


# In[489]:


model.get_weights()


# In[492]:


model.get_weights()


# In[1]:


from keras.utils.vis_utils import plot_model
import graphviz
def _check_pydot():
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
    except Exception:
        # pydot raises a generic Exception here,
        # so no specific class can be caught.
        raise ImportError('Failed to import pydot. You must install pydot'
                          ' and graphviz for `pydotprint` to work.')
plot_model(model, show_shapes=True, show_layer_names=True)


# In[429]:


reportRegressor(model,X_cross,y_cross,X_test,y_test)


# In[431]:


from sklearn.ensemble import VotingRegressor
vc = VotingRegressor(estimators=[('rf',rfr) , ('kn',knn) , ('dt',decisionTree)])
vc.fit(X_train,y_train)
reportRegressor(vc,X_cross,y_cross,X_test,y_test)


# In[ ]:





# In[ ]:





# In[250]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[336]:


inputs = keras.Input(shape=(20,))
dense_layer = layers.Dense(64, activation="relu")
x = dense_layer(inputs)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="SDL_Model")


# In[352]:


model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['mae'])


# In[355]:


hist = model.fit(X_train, y_train,batch_size=64,epochs=200,validation_split=0.2,verbose=1)


# In[356]:


reportRegressor(model,X_cross,y_cross,X_test,y_test)


# In[584]:


acc = []
preds = vc.predict(X_test)
for i in range(len(preds)):
    if (abs(preds[i] - list(y_test)[i]) < 10):
        acc.append('True')
    else:
        acc.append('False')
acc.count('True')/len(acc)


# In[508]:


acc = []
preds = vc.predict(X_test)
for i in range(len(preds)):
    if (abs(preds[i] - list(y_test)[i]) < 100):
        acc.append('True')
    else:
        acc.append('False')
acc.count('True')/len(acc)


# In[509]:


acc = []
preds = vc.predict(X_test)
for i in range(len(preds)):
    if (abs(preds[i] - list(y_test)[i]) < 150):
        acc.append('True')
    else:
        acc.append('False')
acc.count('True')/len(acc)


# In[70]:


def plot_train_test_accuracies(trains,tests,labels):
    fig = plt.figure(figsize=(20,12),frameon=False)
    plt.xlabel('Model Type')
    plt.ylabel('Mean Squared Error')
    plt.axis('off')
    ax = fig.subplots()
    ax.yaxis.grid()
    ax.bar(np.arange(7) - 0.1, [6744.866,2955.728,2429.794,2448.570,2568.890,2590.800,2436.308] , 0.2 , label='Train' , color='salmon')
    ax.bar(np.arange(7) + 0.1, [7926.882,3332.318,3198.896,3196.457,3444.375,3492.698,3056.639] , 0.2 , label='Test' , color='c')
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(['Linear Regressor','SVM','Decision Tree','Random Forest','Nearest Neighbor','Neural Network',' Voting Model'])
    ax.set_title("MSE for each Model (train + test)")
    ax.legend()


# In[71]:


plot_train_test_accuracies(1,1,2)


# In[68]:


def plot_train_test_accuracies(trains,tests,labels):
    fig = plt.figure(figsize=(20,12),frameon=False)
    plt.xlabel('Model Type')
    plt.ylabel('Mean Squared Error')
    plt.axis('off')
    ax = fig.subplots()
    ax.yaxis.grid()
    ax.bar(np.arange(7) - 0.1, [0.453,0.760,0.803,0.802,0.792,0.790,0.803] , 0.2 , label='Train' , color='salmon')
    ax.bar(np.arange(7) + 0.1, [0.433,0.762,0.771,0.771,0.754,0.750,0.781] , 0.2 , label='Test' , color='c')
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(['Linear Regressor','SVM','Decision Tree','Random Forest','Nearest Neighbor','Neural Network',' Voting Model'])
    ax.set_title("R2-Score for each Model (train + test)")
    ax.legend()


# In[69]:


plot_train_test_accuracies(1,1,2)


# In[ ]:


def plot_train_test_accuracies(trains,tests,labels):
    fig = plt.figure(figsize=(20,12),frameon=False)
    plt.xlabel('Model Type')
    plt.ylabel('Mean Squared Error')
    plt.axis('off')
    ax = fig.subplots()
    ax.yaxis.grid()
    ax.bar(np.arange(7) - 0.1, [0.453,0.760,0.803,0.802,0.792,0.790,0.803] , 0.2 , label='Train' , color='salmon')
    ax.bar(np.arange(7) + 0.1, [0.433,0.762,0.771,0.771,0.754,0.750,0.781] , 0.2 , label='Test' , color='c')
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(range(300))
    ax.set_title("R2-Score for each Model (train + test)")
    ax.legend()


# In[146]:


y_test = list(y_test)


# In[144]:


preds = vc.predict(X_test)
preds


# In[163]:


len(preds)


# In[149]:


y_test[1]


# In[155]:


d = pd.DataFrame(columns=['Model Prediction','Real Value'])


# In[157]:


d['Model Prediction'] = preds
d['Real Value'] = y_test


# In[162]:


sns.set(style="darkgrid")

tips = sns.load_dataset("tips")
g = sns.jointplot("Model Prediction", "Real Value", data=d,
                  kind="reg", truncate=False,
                  xlim=(0, 320), ylim=(0, 320),
                  color="m", height=7)

