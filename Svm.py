import getdata as gd
import numpy as np
from sklearn.svm import SVC
import pickle
clf4=SVC(kernel='linear')
clf4.fit(gd.features_train,gd.label_train)
path=open('svm.pkl','wb')
pickle.dump(clf4,path)
path.close()
