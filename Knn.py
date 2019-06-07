import getdata as gd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
clf2=KNeighborsClassifier(n_neighbors=7,weights='distance')
clf2.fit(gd.features_train,gd.label_train)
path=open('knn.pkl','wb')
pickle.dump(clf2,path)
path.close()
