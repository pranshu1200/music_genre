import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
features=[]
label=[]
with open('data2.csv','r') as csvFile:
	reader=csv.reader(csvFile)
	j=0
	for row in reader:
		x=[]
		if(j!=0):	
			for k in row[1:len(row)-1]:
				x.append((float)(k))
			features.append(x)
			label.append(row[-1])
		j=j+1
features_train,features_test,label_train,label_test=train_test_split(features,label,test_size=0.4)
