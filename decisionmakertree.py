#this code will create a music_recommender.dot file in the working directory open it with visual code(download the extension Graphviz(dot)language then select open preview to side)
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree #imports the decision tree in the graphical format 
#import the dataset
music_data = pd.read_csv('music.csv')
#creating input set 2 columns - age and gender 
X = music_data.drop(columns=['genre'])
#creating the output set containing the gender
y = music_data['genre']
#create a model
model = DecisionTreeClassifier()
#train the model
model.fit(X,y)
tree.export_graphviz(model, out_file='music_recommender.dot',
                    feature_names=['age', 'gender'],
                     class_names=sorted(y.unique()),#to display the name of the genre like hiphop, classical etc
                     label='all',
                     rounded=True,#corners of each node is rounded
                     filled=True)#each box or node (graph) is filled with a color

