import getdata as gt
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
import numpy as np
path=open('svm.pkl','rb')
clf=pickle.load(path)
y=clf.predict(gt.features_test)
print(accuracy_score(gt.label_test,y))
print(confusion_matrix(gt.label_test,y))
