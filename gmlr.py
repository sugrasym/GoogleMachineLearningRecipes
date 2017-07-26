#MAchine Learning #1

import sklearn
"""Training data where the first number is the weight and second is whether its skin is bumpy (1) or smooth (0)"""
features = [[140, 1], [130, 1], [150, 0], [170,0]]

labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print clf.predict([[150,0]])

#Machine Learning #2

from sklearn.datasets import load_iris

iris = load_iris()

print iris.feature_names

print iris.target_names

#The below code will print given information for the first item in the list
print iris.data[0]

print iris.target[0]
#THe code below is used to print the entire table of information instead of just line
for i in range(len(iris.target)):
	print "Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])

#The following data is to test the machine against data it might not have seen yet
import numpy as np
from sklearn.datasets import load_iris
from skleanr import tree
iris = load_iris()
test_idc = [0,50,100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = np.delete(iris.target, test_idx)
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)

"""viz code (http://scikit-learn.org/stable/modules/tree.html#tree "Ctrl+F 'if we'to find the code)"""

>>> import pydotplus 
>>> dot_data = tree.export_graphviz(clf, out_file=None) 
>>> graph = pydotplus.graph_from_dot_data(dot_data) 
>>> graph.write_pdf("iris.pdf") 

print test_data[0], test_target[0]

print iris.feature_names, iris.target_names
