import tensorflow as tf
from tensorflow import keras
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


#Code One-Hot Encodes input

number0 = ' '
number1 = '0123456789'


class OneHotEncoding():

	def encode(self,dataList):
		tempData = dataList.copy()
		self.padData(tempData)
		output = []
		for i in range(0,len(tempData)):
			word = tempData[i]
			oneHot = []
			for j in range(0,len(word)):
				if(word[j] in number0):
					oneHot.append(np.array([1,0,0]))
				elif(word[j] in number1):
					oneHot.append(np.array([0,1,0]))
				else:
					oneHot.append(np.array([0,0,1]))
			output.append(np.array(oneHot))
		return np.array(output)

	def padData(self,dataList):
		maxLen = 0
		for add in dataList:
			if(len(add)>=maxLen):
				maxLen = len(add)

		for i in range(0,len(dataList)):
			dataList[i] = dataList[i].ljust(maxLen)








