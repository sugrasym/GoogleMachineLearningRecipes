#MAchine Learning #1

"""Code that has been commented out is more than likely added to the exisiting code eventually"""

import sklearn
"""Training data where the first number is the weight and second is whether its skin is bumpy (1) or smooth (0)"""
features = [[140, 1], [130, 1], [150, 0], [170,0]]

labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print clf.predict([[150,0]])

#Machine Learning #2

from sklearn.datasets import load_iris

#iris: "https://en.wikipedia.org/wiki/Iris_flower_data_set:"	
iris = load_iris()

print iris.feature_names

print iris.target_names

#The below code will print given information for the first item in the list
print iris.data[0]

print iris.target[0]
#The code below is used to print the entire table of information instead of just 1 line

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

#Machine Learning #3

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500
#The code below will generate a height for each dog randomly within 4in of the avg. height
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show() 

#Machine Learning #4

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target
from sk.learn.cross_validation import train_test_split
X_train, X_test, y_train, y_tst = train_test_split(X, y, test_size = .5)

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

"""from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNneighborsClassifier()"""

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print predictions

"""from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)"""

#Machine Learning #5