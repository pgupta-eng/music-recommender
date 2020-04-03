import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier  #sklearn is the package & algoritm udes is decision tree
from sklearn.externals import joblib #used for saving and loading models
music_data = pd.read_csv('music.csv')
#music_data #in the csv file 1 is given for male and 0 is given for female 
#we will now give split the dataset into input set and output set: i/p user is 21 years old and is male output set: what is the genre of music he might like
#we do not want to train ourr module every time therefore commented
X = music_data.drop( columns=['genre'])#returns input data set
# #.drop creates another dataset without the specified column
# X
y = music_data['genre']#returns output data set#algorithm used is: decision tree

model = DecisionTreeClassifier()
model.fit(X,y)
#model = joblib.load('music-recommender.joblib')#stores the name of the file where in we are saving the trained data set
predictions = model.predict(np.array([[21, 1]]))
predictions