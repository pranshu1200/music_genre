import getdata as gd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
clf5=RandomForestClassifier(min_samples_split=5,n_estimators=15,criterion='entropy')
clf5.fit(gd.features_train,gd.label_train)
path=open('randomforest.pkl','wb')
pickle.dump(clf5,path)
path.close()
