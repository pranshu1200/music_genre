# music_genre
music genre classification using machine learning algorithms

This is a project that works to classify the music into one of the 10 genres which are 'blues','classical','country','disco','hiphop','jazz','metal','pop','reggae' and 'rock'.
The project uses models trained using the following Algorithms:
  GaussianNB
  SVM with linear kernel
  SVM with rbf kernel
  Decision Tree Classifier
  Random Forest Classifier
  K-Nearest Neighbors
  
The dataset used is GTZAN dataset consisting of 1000 music clips of 30 seconds each (100 for each genre) in .au format and can be found here http://marsyas.info/downloads/datasets.html
The clips were changed to .wav format using sox tool on linux.

The features were extracted out of the clips using lebrosa module of python and saved into .csv file (data2.csv) using the file 'make_csv.py'.

The file 'test_new.py' contains the code to classify the genre of any new song by its path from the available models.

Information regarding accuracy and confusion matrix for each model can be found in the file 'results' in the repository itself.
