import tensorflow as tf
from tensorflow import keras
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

#Finds split locations of addresses and writes to Database

pd.set_option('display.max_columns', 1000)

data = pd.read_csv('someFileHere',encoding = 'utf8')

print(len(list(data)))
headers = ['NUMBER','STREET','CITY','REGION','POSTCODE']
headers2 = ['NUMBER','STREET','CITY', 'POSTCODE']

indexArrays = []


if (len(list(data))-1 == len(headers)):
   for i in range(0,len(data)):
    startIndex = 0
    tempArray = []
    tempArray.append(startIndex)
    for j in range(2):
        startIndex += len(str(data.loc[i,headers[j]]))
    startIndex +=1
    tempArray.append(startIndex)
    for k in range(2,4):
        startIndex += len(str(data.loc[i,headers[k]]))
    startIndex +=2
    tempArray.append(startIndex)
    for m in range(4,5):
        startIndex += len(str(data.loc[i,headers[m]]))
        startIndex +=1
        tempArray.append(startIndex)
    indexArrays.append(tempArray)
else:
    for i in range(0,len(data)):
        startIndex = 0
        tempArray = []
        tempArray.append(startIndex)
        for j in range(2):
            startIndex += len(str(data.loc[i,headers2[j]]))
        startIndex +=1
        tempArray.append(startIndex)
        for k in range(2,3):
            startIndex += len(str(data.loc[i,headers2[k]]))
            tempArray.append(startIndex)
        for m in range(3,4):
            startIndex += len(str(data.loc[i,headers2[m]]))
            startIndex+=2
            tempArray.append(startIndex)
        indexArrays.append(tempArray)


returningData = pd.DataFrame(np.array(indexArrays),columns = ['index0','index1','index2','index3'])

returningData['ADDRESS'] = data.ADDRESS

print(returningData[1:10])

conn = sqlite3.connect('addressData.db')
returningData.to_sql('addresses', con=conn,if_exists = "append")