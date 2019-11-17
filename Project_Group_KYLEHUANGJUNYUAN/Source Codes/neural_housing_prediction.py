#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pylab as plt
import pandas as pd 
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from subprocess import check_output
from math import sqrt


# In[2]:


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[3]:


NUM_FEATURES = 49
NUM_OUTPUTS = 1

learning_rate = 0.001
epochs = 500
batch_size = 64
beta = 0.001
num_hidden_layer_neuron = 10
seed = 100
np.random.seed(seed)


# In[4]:


def extract_feature(col_names, data):

    data = data[col_names[1:]]
    return data


# In[5]:


def feature_discovery(data,output):
    data = pd.concat([data,output],sort=False, axis=1)
    redundunt_feature_to_drop = ['GarageCars', '1stFlrSF','YearRemodAdd','BsmtQual_Ex','TotRmsAbvGrd']
    data.drop(redundunt_feature_to_drop,axis=1,inplace=True)
    correlationMatrix = data.corr()
    k = 50
    cols = correlationMatrix.nlargest(k, 'SalePrice')['SalePrice'].index

    return cols


# In[6]:


def dropped_redundunt_features(data):

    #find columns with high percentage of zeroes
    find_zeroes = data.columns[(data == 0).any()]
    num = (data[find_zeroes] == 0).sum()
    per_zeros = round(num/data.shape[0],2)
    
    #find features that has more than 90% zeroes
    features_with_zeroes = per_zeros[per_zeros>=0.80].index
    
    #Remove columns
    data.drop(features_with_zeroes,axis=1,inplace=True)

    #find columns with high percentage of NA
    find_na = data.isna().any()
    columns_na = data[data.columns[find_na]]
    na = columns_na.isna().sum()
    per_na = round(na/data.shape[0],2)

    #find features that has more than 90% NA
    features_with_NA = per_na[per_na>=0.80].index
    
    #Remove columns
    data.drop(features_with_NA,axis=1,inplace=True)

    return data


# In[7]:


def removeRowsWithNA(data):
    #Replace Numerical NA with average value
    data = data.fillna(data.mean().to_dict())

    #Replace String NA with 'None'
    data = data.fillna("None")

    return data


# In[8]:


def replaceNARows(data):

    #Replace Numerical NA with average value
    data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
    data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(data['BsmtFinSF1'].mean())
    # data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(data['BsmtFinSF2'].mean())
    data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(data['BsmtUnfSF'].mean())
    data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(data['TotalBsmtSF'].mean())
    data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].median())
    data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)
    data['GarageCars'] = data['GarageCars'].fillna(0)
    data['GarageArea'] = data['GarageArea'].fillna(0)
    data['BsmtFullBath'] = data['BsmtFullBath'].fillna(0)

    #Replace String NA with 'None'
    data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
    data['Utilities'] = data['Utilities'].fillna(data['Utilities'].mode()[0])
    data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
    data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
    data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
    data['Functional'] = data['Functional'].fillna(data['Functional'].mode()[0])
    data['SaleType'] =data['SaleType'].fillna(data['SaleType'].mode()[0])

    data['MasVnrType'] = data['MasVnrType'].fillna('None')
    data['BsmtQual'] = data['BsmtQual'].fillna('None')
    data['BsmtCond'] = data['BsmtCond'].fillna('None')
    data['BsmtExposure'] = data['BsmtExposure'].fillna('None')
    data['BsmtFinType1'] = data['BsmtFinType1'].fillna('None')
    data['BsmtFinType2'] = data['BsmtFinType2'].fillna('None')
    data['FireplaceQu'] = data['FireplaceQu'].fillna('None')
    data['GarageType'] = data['GarageType'].fillna('None')
    data['GarageFinish'] = data['GarageFinish'].fillna('None')
    data['GarageQual'] = data['GarageQual'].fillna('None')
    data['GarageCond'] = data['GarageCond'].fillna('None')
    # data['Fence'] = data['Fence'].fillna('None')

    return data



# In[10]:


#Retrive datasets
trainX= pd.read_csv('trainX.csv')
trainY= pd.read_csv('trainY.csv') /10000

testX = pd.read_csv('testX.csv')
testY = pd.read_csv('testY.csv') /10000

#drop redundunt feature 
combineX = pd.concat([trainX,testX],sort=False).reset_index(drop=True)
combineX = dropped_redundunt_features(combineX)

#replace NA values
combineX = replaceNARows(combineX)

combineX = pd.get_dummies(combineX)



# In[11]:


#Normalize encoded data
normalized_combineX = (combineX - np.mean(combineX, axis=0))/np.std(combineX,axis=0)

num_of_train = trainX.shape[0]
trainX_new = normalized_combineX.iloc[:num_of_train,:]
testX_new = normalized_combineX.iloc[num_of_train:,:]

#Get correlation after normalization
features = feature_discovery(trainX_new, trainY)

trainX = extract_feature(features, trainX_new)
testX = extract_feature(features, testX_new)



# In[13]:


# changing to matrix for input neural network
trainX = trainX.values
testX = testX.values 

trainY = trainY.values
testY = testY.values
trainY1 = trainY.reshape(trainY.shape[0],1)


# In[14]:


#Shuffling the training data
idx = np.arange(trainX.shape[0]) 
np.random.shuffle(idx) #shuffling the index 
trainX, trainY = trainX[idx], trainY[idx] #Giving the data with shuffled indexes
data_len = len(trainX)



# In[25]:


num_data = len(trainX)



# In[26]:


def model_3_layer(trainX, trainY, testX, testY, num_data ,NUM_FEATURES ,learning_rate = 0.001, epochs = 200, batch_size = 64, num_hidden_layer_neuron = 50, beta = 0.001, plot_test = True, dropout = False):

    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_OUTPUTS])

    # Build the graph for deep net 3 layers
    weights_1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_hidden_layer_neuron], stddev=1.0 / np.sqrt(float(NUM_FEATURES)), dtype=tf.float32), name='weights_1')
    
    biases_1 = tf.Variable(tf.zeros([num_hidden_layer_neuron]),dtype=tf.float32 ,name='biases_1')
    
    weights_2 = tf.Variable(tf.truncated_normal([num_hidden_layer_neuron, NUM_OUTPUTS], stddev=1.0 / np.sqrt(float(num_hidden_layer_neuron)),dtype=tf.float32), name='weights_2')
    
    biases_2 = tf.Variable(tf.zeros([NUM_OUTPUTS]), dtype=tf.float32,name='biases_2')

    # Forward propogation 
    I1 = tf.matmul(x,weights_1) + biases_1
    H1 = tf.nn.relu(I1)
    if dropout:
        H1 = tf.nn.dropout(H1, keep_prob = 0.8)
    output_y = tf.matmul(H1,weights_2) + biases_2
    if dropout:
        output_y = tf.nn.dropout(output_y, keep_prob = 0.8)

    y = output_y # linear output

    error = tf.reduce_mean(tf.square(y_ - y)) #Normal loss function

    # Loss function with L2 Regularization with beta = 0.001
    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
    loss = tf.reduce_mean(error + beta * regularizers)

    # Optimizer
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err = []
        test_err = []
        predicted_value = []
        idx = np.arange(num_data)
        len_batch_size = np.ceil(num_data/batch_size)
         
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX = trainX[idx]
            trainY = trainY[idx]

            batch_loss = 0
            # Created mini batches for training during epochs
            for start, end in zip(range(0, num_data, batch_size), range(batch_size, num_data, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
                batch_loss += loss.eval(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
                
                
            train_err.append(batch_loss/len_batch_size)
            

            prediction = np.squeeze(y.eval(feed_dict={x: testX}))
        
            epoch_test_error_ = np.mean(np.square(prediction - testY))
            
            test_err.append(epoch_test_error_)

            if i % 50 == 0:
                print('iter %d: train error %g'%(i, train_err[i]))
                print('iter %d: test error %g'%(i, test_err[i]))
            if i == 199:
                print('At 200 iter: test error at this point is %g'%(test_err[i]))
            if i == 499:
                print('At 500 iter: test error at this point is %g'%(test_err[i]))
    
        predicted_value = y.eval(feed_dict={x: testX})
        
        if plot_test:
            # plot test error and predicted values
            plt.figure(3)
#             plt.scatter(range(testX.shape[0]), predicted_value[:testX.shape[0]], color = 'red')
#             plt.scatter(range(testX.shape[0]), Y_test[:testX.shape[0]])
            plt.plot(range(80), predicted_value[:80], label = 'prediction',color = 'red')
            plt.plot(range(80), testY[:80], label = 'actual', color='orange')
            plt.xlabel('Samples')
            plt.ylabel('Values')
            plt.legend()
            plt.savefig('plot test(Nelson).png')
            plt.show()

        return train_err, test_err, predicted_value


# In[27]:


def model_4_layers( trainX, trainY, testX, testY, num_data ,NUM_FEATURES ,learning_rate = 0.001, epochs = 200, batch_size = 64, num_hidden_layer_neuron = [500,500], beta = 0.001, plot_test = True, dropout = False):
        
     prob = 0.8
                   
     x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
     y_ = tf.placeholder(tf.float32, [None, NUM_OUTPUTS])

     keep_prob = tf.placeholder(tf.float32)

     # Build the graph for deep net 4 layers
     weights_1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_hidden_layer_neuron[0]], stddev=1.0 / np.sqrt(float(NUM_FEATURES)), dtype=tf.float32), name='weights_1')
    
     biases_1 = tf.Variable(tf.zeros([num_hidden_layer_neuron[0]]),dtype=tf.float32 ,name='biases_1')
    
     weights_2 = tf.Variable(tf.truncated_normal([num_hidden_layer_neuron[0], num_hidden_layer_neuron[1]], stddev=1.0 / np.sqrt(float(num_hidden_layer_neuron[0])),dtype=tf.float32), name='weights_2')
    
     biases_2 = tf.Variable(tf.zeros(num_hidden_layer_neuron[1]), dtype=tf.float32,name='biases_2')

     weights_3 = tf.Variable(tf.truncated_normal([num_hidden_layer_neuron[1], NUM_OUTPUTS], stddev=1.0 / np.sqrt(float(num_hidden_layer_neuron[1])),dtype=tf.float32), name='weights_3')
    
     biases_3 = tf.Variable(tf.zeros([NUM_OUTPUTS]), dtype=tf.float32,name='biases_3')

     # Forward propogation 4 layers
     M1 = tf.matmul(x,weights_1) + biases_1
     H1 = tf.nn.relu(M1)
     if dropout:
         H1 = tf.nn.dropout(H1, keep_prob)

     M2 = tf.matmul(H1,weights_2) + biases_2
     H2 = tf.nn.relu(M2)
     if dropout:
         H2 = tf.nn.dropout(H2, keep_prob)
     
     M3 = tf.matmul(H2,weights_3) + biases_3
     if dropout:
         M3 = tf.nn.dropout(M3, keep_prob)

     y = M3 # linear output

     error = tf.reduce_mean(tf.square(y_ - y)) #Normal loss function

     # Loss function with L2 Regularization with beta = 0.001
     regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3)
     loss = tf.reduce_mean(error + beta * regularizers)

     # Optimizer
     train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

     with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err = []
        test_err = []
        predicted_value = []
        idx = np.arange(num_data)
        len_batch_size = np.ceil(num_data/batch_size)
         
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX = trainX[idx]
            trainY = trainY[idx]

            batch_loss = 0
            # Created mini batches for training during epochs
            for start, end in zip(range(0, num_data, batch_size), range(batch_size, num_data, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob : prob})
                batch_loss += loss.eval(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob:prob})
                
                
            train_err.append(batch_loss/len_batch_size)
            

            prediction = np.squeeze(y.eval(feed_dict={x: testX , keep_prob: 1.0}))
        
            epoch_test_error_ = np.mean(np.square(prediction - testY))
            
            test_err.append(epoch_test_error_)

            if i % 10 == 0:
                print('iter %d: train error %g'%(i, train_err[i]))
                print('iter %d: test error %g'%(i, test_err[i]))
            
            if i == 149:
                print('iter %d: train error %g'%(i, train_err[i]))
                print('iter %d: test error %g'%(i, test_err[i]))
           
            if i == 499:
                    print('At 499 iter: 4 layer train error at this point is %g'%(train_err[i]))
                    print('At 499 iter: 4 layer test error at this point is %g'%(test_err[i]))
            if i == 999:
                    print('At 999 iter: 4 layer train error at this point is %g'%(train_err[i]))

        predicted_value = y.eval(feed_dict={x: testX,keep_prob : 1.0})
        
        if plot_test:
            # plot test error and predicted values
            plt.figure(3,figsize=(14,7))
            plt.scatter(range(50), predicted_value[:50])
            plt.plot(range(50), predicted_value[:50], label = 'prediction')
            plt.scatter(range(50), testY[:50])
            plt.plot(range(50), testY[:50], label = 'actual')
            plt.xlabel('Samples')
            plt.ylabel('Values')
            plt.legend()
            plt.show()

        return train_err, test_err, predicted_value


# In[28]:


def model_5_layers(trainX, trainY, testX, testY, num_data ,NUM_FEATURES ,learning_rate = 0.001, epochs = 200, batch_size = 64, num_hidden_layer_neuron = [100,100,100], beta = 0.001, plot_test = True, dropout = False):
                   
     prob = 0.8
     
     x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
     y_ = tf.placeholder(tf.float32, [None, NUM_OUTPUTS])
     keep_prob = tf.placeholder(tf.float32)

     # Build the graph for deep net 5 layers
     weights_1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_hidden_layer_neuron[0]], stddev=1.0 / np.sqrt(float(NUM_FEATURES)), dtype=tf.float32), name='weights_1')
    
     biases_1 = tf.Variable(tf.zeros([num_hidden_layer_neuron[0]]),dtype=tf.float32 ,name='biases_1')
    
     weights_2 = tf.Variable(tf.truncated_normal([num_hidden_layer_neuron[0], num_hidden_layer_neuron[1]], stddev=1.0 / np.sqrt(float(num_hidden_layer_neuron[0])),dtype=tf.float32), name='weights_2')
    
     biases_2 = tf.Variable(tf.zeros(num_hidden_layer_neuron[1]), dtype=tf.float32,name='biases_2')

     weights_3 = tf.Variable(tf.truncated_normal([num_hidden_layer_neuron[1], num_hidden_layer_neuron[2]], stddev=1.0 / np.sqrt(float(num_hidden_layer_neuron[1])),dtype=tf.float32), name='weights_3')
    
     biases_3 = tf.Variable(tf.zeros([num_hidden_layer_neuron[2]]), dtype=tf.float32,name='biases_3')

     weights_4 = tf.Variable(tf.truncated_normal([num_hidden_layer_neuron[2], NUM_OUTPUTS], stddev=1.0 / np.sqrt(float(num_hidden_layer_neuron[2])),dtype=tf.float32), name='weights_4')
    
     biases_4 = tf.Variable(tf.zeros([NUM_OUTPUTS]), dtype=tf.float32,name='biases_4')

     # Forward propogation 5 layers
     M1 = tf.matmul(x,weights_1) + biases_1
     H1 = tf.nn.relu(M1)
     if dropout:
         H1 = tf.nn.dropout(H1, keep_prob)

     M2 = tf.matmul(H1,weights_2) + biases_2
     H2 = tf.nn.relu(M2)
     if dropout:
         H2 = tf.nn.dropout(H2, keep_prob)
     
     M3 = tf.matmul(H2,weights_3) + biases_3
     H3 = tf.nn.relu(M3)
     if dropout:
         H3 = tf.nn.dropout(H3, keep_prob)

     M4 = tf.matmul(H3,weights_4) + biases_4
     if dropout:
         M4 = tf.nn.dropout(M4, keep_prob)

     y = M4   # linear output

     error = tf.reduce_mean(tf.square(y_ - y)) #Normal loss function

     # Loss function with L2 Regularization with beta = 0.001
     regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4)
     loss = tf.reduce_mean(error + beta * regularizers)

     # Optimizer
     train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

     with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_err = []
        test_err = []
        predicted_value = []
        idx = np.arange(num_data)
        len_batch_size = np.ceil(num_data/batch_size)
         
        for i in range(epochs):
            np.random.shuffle(idx)
            trainX = trainX[idx]
            trainY = trainY[idx]

            batch_loss = 0
            # Created mini batches for training during epochs
            for start, end in zip(range(0, num_data, batch_size), range(batch_size, num_data, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob:prob})
                batch_loss += loss.eval(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob:prob})
                
                
            train_err.append(batch_loss/len_batch_size)

            prediction = np.squeeze(y.eval(feed_dict={x: testX, keep_prob : 1.0}))
        
            epoch_test_error_ = np.mean(np.square(prediction - testY))
            
            test_err.append(epoch_test_error_)

            if i % 50 == 0:
                print('iter %d: train error %g'%(i, train_err[i]))
                print('iter %d: test error %g'%(i, test_err[i]))
            if i == 99:
                print('iter %d: train error %g'%(i, train_err[i]))
                print('iter %d: test error %g'%(i, test_err[i]))
            if i == 499:
                print('iter %d: train error %g'%(i, train_err[i]))
        
        
        predicted_value = y.eval(feed_dict={x: testX, keep_prob : 1.0})
        
        if plot_test:
            # plot test error and predicted values
            plt.figure(3)
            plt.scatter(range(50), predicted_value[:50])
            plt.plot(range(50), predicted_value[:50], label = 'prediction')
            plt.scatter(range(50), testY[:50])
            plt.plot(range(50), testY[:50], label = 'actual')
            plt.xlabel('Samples')
            plt.ylabel('Values')
            plt.legend()
            plt.show()

        return train_err, test_err, predicted_value


# In[29]:


def plot_learning_curve(epochs, train_err, test_err, figname):
    # plot learning curves
    plt.figure(1)
    plt.plot(range(epochs), train_err , label = 'Training error')
    plt.plot(range(epochs), test_err, label = 'Test error')
    plt.legend()
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Errors')
    plt.title(figname)
    plt.savefig(figname)
    plt.show()


# In[30]:


def rmsle(y, y_pred):
  y = y * 10000
  y_pred = y_pred * 10000
  assert len(y) == len(y_pred)
  terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
  return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


# In[59]:


def plotRegressionGraph(prediction,test_output):
    prediction = prediction * 10000
    test_output = test_output * 10000
    fig, ax = plt.subplots(figsize=(11,7))
    ax.scatter(prediction, test_output)
    ax.plot([prediction.min(),prediction.max()],[prediction.min(),prediction.max()], 'k--', lw=3, color='black')
    ax.set_xlabel('Predicted Sale Price')
    ax.set_ylabel('Actual Sale Price')
    plt.tight_layout(pad=1)
    plt.savefig('NN Regression Graph')
    plt.show()


# In[32]:


def plot_graph(Y_test, Y_test_predict, filename):
    Y_test = Y_test * 10000
    Y_test_predict = Y_test_predict * 10000
    df = pd.DataFrame({'Actual': Y_test.flatten(),'Predicted': Y_test_predict.flatten()})
    df1 = df.head(30)
    df1.plot(kind='bar',figsize=(16,9))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.xticks(rotation=0)
    plt.xlabel('Test Sample')
    plt.ylabel('SalePrice')
    plt.savefig(filename)
    plt.legend()
    plt.show()


# In[33]:


if __name__ == "__main__":
    train_err_1, test_err_1, predicted_value = model_4_layers(trainX, trainY, testX, testY, num_data ,49, epochs = 32,plot_test = True, dropout = False)
    plot_learning_curve(32, train_err_1,test_err_1,'Learning curve 4 layer testing optimum NN')
    plot_graph(testY, predicted_value, 'Actual vs Predicted')
    log_error = rmsle(testY,predicted_value)
    print('Root mean log error: ', log_error)
    
    

## Trying other function from github


# In[60]:


plotRegressionGraph(predicted_value, testY)


# In[37]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
mse = mean_squared_error(testY*10000, predicted_value*10000)
mae = mean_absolute_error(testY*10000, predicted_value*10000)
rmse = sqrt(mean_squared_error(testY*10000, predicted_value*10000)) 
print('Mean Absolute Error = '+str(mae))
print('Mean Square Error = '+str(mse))
print('Root Mean Square Error = ' +str(rmse))


# In[38]:


def rmse(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))*100

    return loss


# In[39]:


rmspe = rmse(testY*10000, predicted_value*10000)
mape = np.mean(np.abs((testY*10000) - (predicted_value*10000)) / (testY*10000)) * 100
print("MAPE = %.2f%%" % mape)
print("RMSPE = %.2f%%" % rmspe)




