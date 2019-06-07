import pickle
import os
import librosa
import numpy as np
path=open('randomforest.pkl','rb')
clf=pickle.load(path)
music=f'/home/pranshu/Desktop/genres/rock/rock.00040.wav'
l=[]
y,sr=librosa.load(music,mono=True,duration=120)
l.append(np.mean(librosa.feature.chroma_stft(y=y,sr=sr)))
l.append(np.mean(librosa.feature.mfcc(y=y, sr=sr)))
l.append(np.mean(librosa.feature.rmse(y=y)))
l.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
l.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
l.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
l.append(np.mean(librosa.feature.zero_crossing_rate(y)))
print(clf.predict([l]))
