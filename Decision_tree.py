import getdata as gd
import numpy as np
from sklearn import tree
import pickle
clf3=tree.DecisionTreeClassifier(max_depth=1000)
clf3.fit(gd.features_train,gd.label_train)
path=open('decisiontree.pkl','wb')
pickle.dump(clf3,path)
path.close()
