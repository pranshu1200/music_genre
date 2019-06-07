import librosa
import numpy as np
import os
import pandas as pd
import csv
genres='blues classical country disco hiphop jazz metal pop reggae rock'.split()
list_of_data=[]
cl=0
for g in genres:
	for filename in os.listdir(f'/home/pranshu/Desktop/genres/{g}'):
		d={}
		music=f'/home/pranshu/Desktop/genres/{g}/{filename}'
		y,sr=librosa.load(music,mono=True,duration=30)
		d['chrom_stft']=np.mean(librosa.feature.chroma_stft(y=y,sr=sr))
		#d['tonnetz']=np.mean(librosa.feature.tonnetz(y=y,sr=sr))
		d['rmse']=np.mean(librosa.feature.rmse(y=y))
		d['spec_cent']=np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
		d['spec_bw']=np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
		d['rolloff']=np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
		d['zcr']=np.mean(librosa.feature.zero_crossing_rate(y))
		d['mfcc']=np.mean(librosa.feature.mfcc(y=y, sr=sr))
		d['genre']=g
		list_of_data.append(d)
		print(cl)
		cl=cl+1
df=pd.DataFrame(list_of_data)
df.to_csv('data2.csv')		
		
