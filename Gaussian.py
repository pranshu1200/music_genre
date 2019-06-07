import getdata as gd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pickle
clf1=GaussianNB()
clf1.fit(gd.features_train,gd.label_train)
path=open('guassiannb.pkl','wb')
pickle.dump(clf1,path)
path.close()
