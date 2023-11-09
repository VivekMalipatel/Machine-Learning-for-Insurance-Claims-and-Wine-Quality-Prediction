import pandas
import sklearn.tree as tree
from sklearn.metrics import accuracy_score
import numpy

trainData = pandas.read_csv('WineQuality_Train.csv', delimiter=',')
nObs = trainData.shape[0]

x_train = trainData[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = trainData['quality_grp']

ensemblePredProb = numpy.zeros((nObs, 2))
w_train = numpy.full(nObs, 1.0)

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)
treeFit = classTree.fit(x_train, y_train, w_train)
treePredProb = classTree.predict_proba(x_train)
y_predClass = numpy.where(treePredProb[:,1] >= 0.2, 1, 0)
accuracy = numpy.sum(numpy.where(y_train == y_predClass, w_train, 0.0)) / numpy.sum(w_train)
ensemblePredProb += accuracy * treePredProb
ensemblePredProb /= numpy.sum(accuracy)

trainData['prediction_quality_grp'] = numpy.where(ensemblePredProb[:,1] >= 0.2, 1, 0)
ensembleAccuracy = numpy.mean(numpy.where(trainData['prediction_quality_grp'] == y_train, 1, 0))
print("The Misclassification Rate of the classification tree on the Training data at Iteration 0 : ",1-ensembleAccuracy)


'''

testData = pandas.read_csv('WineQuality_Test.csv', delimiter=',', usecols=['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates','quality_grp'])

x_test = testData[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_test = testData['quality_grp']

testtreePredProb = classTree.predict_proba(x_test)
ytest_predClass = numpy.where(testtreePredProb[:,1] >= 0.2, 1, 0)
print(accuracy_score(y_test,ytest_predClass))
'''