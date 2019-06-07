import getdata as gd
import numpy as np
from sklearn.svm import SVC
import pickle
clf6=SVC(kernel='rbf')
clf6.fit(gd.features_train,gd.label_train)
path=open('svm2.pkl','wb')
pickle.dump(clf6,path)
path.close()
