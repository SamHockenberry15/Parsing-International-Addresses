import tensorflow as tf
from tensorflow import keras
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import DataConversion_v1_1
import sqlite3


def get_indices_from_sql():
    conn = sqlite3.connect("addressData.db")
    df = pd.read_sql_query("select * from addresses;", conn)
    return df

def get_index_arrays(data):
    all_rows = []

    for row in data.iloc[:,2:5].values:
        new_row = np.zeros(50)
        for i in range(3):
            new_row[row[i]] = 1
        all_rows.append(new_row)
    result = np.array(all_rows)
    return result

def accuracy_test(data, true_ans):
    res = np.abs(data - true_ans)
    totalVals = float(len(data[0]) * len(data))
    count = 0.0
    for row in res:
        for num in row:
            if num == 0:
                count+=1.0
    return count/totalVals

np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_columns', None) 

data = get_indices_from_sql()
data = data[data['ADDRESS'].str.len() < 50]


tf.set_random_seed(1)
addressList = data['ADDRESS'].tolist()


newData = DataConversion_v1_1.OneHotEncoding().encode(addressList)
newData = newData.reshape(len(newData),len(newData[0])*len(newData[0][0]))
outputData = get_index_arrays(data)

eightyP = round(len(newData)*.85)
x_train = newData[0:eightyP]

x_test =  newData[eightyP:len(newData)]
y_train = outputData[0:eightyP]
y_test = outputData[eightyP:len(newData)]

hid_nodes = 20
out_nodes = len(outputData[0])

X = tf.placeholder(tf.float32, [None, len(x_train[0])])
Y = tf.placeholder(tf.float32, [None, len(y_train[0])])

w0 = tf.Variable(tf.random_normal([len(x_train[0]),hid_nodes]))
w1 = tf.Variable(tf.random_normal([hid_nodes,out_nodes]))

b0 = tf.Variable(tf.random_normal([hid_nodes]))
b1 = tf.Variable(tf.random_normal([out_nodes]))

layer_1 = tf.add(tf.matmul(X,w0),b0)
layer_1 = tf.nn.relu(layer_1)

out_layer = tf.matmul(layer_1,w1) + b1

sig = tf.nn.sigmoid_cross_entropy_with_logits(logits = out_layer, labels = Y)
loss = tf.reduce_mean(sig)

learning_rate = .01
num_epochs = 35
batch_size = 200

batches = int(y_train.shape[0] / batch_size)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#Neural Network code

with tf.Session() as sesh:
    sesh.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
    	for batch in range(batches):
    		offset = batch * epoch
    		x = x_train[offset: offset + batch_size]
    		y = y_train[offset: offset + batch_size]
    		sesh.run(optimizer,feed_dict = {X:x,Y:y})
    		ls = sesh.run(loss, feed_dict={X:x, Y:y})
    	if not epoch % 2:
    		print('epoch: ',epoch,  'cost= ', ls)


    prediction = tf.equal(tf.argmax(out_layer,1),tf.argmax(Y,1))
    success = tf.reduce_mean(tf.cast(prediction,tf.float32))

    
    t1 = out_layer.eval({X:x_test})

    print('Success Rate: ', sesh.run(success,feed_dict={X:x_test,Y:y_test}))
    print(accuracy_test(np.sort(sesh.run(tf.nn.top_k(tf.nn.sigmoid(t1),3)).indices),np.sort(sesh.run(tf.nn.top_k(y_test,3).indices))))