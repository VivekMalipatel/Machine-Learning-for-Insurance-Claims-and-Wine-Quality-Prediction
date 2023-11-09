import numpy
import pandas
from sklearn import (ensemble, metrics, tree, )
import matplotlib.pyplot as plt

trainData = pandas.read_csv('WineQuality_Train.csv', delimiter=',')
testData = pandas.read_csv('WineQuality_Test.csv', delimiter=',', usecols=['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates','quality_grp'])
nObs = trainData.shape[0]

x_train = trainData[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = trainData['quality_grp']

x_test = testData[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_test = testData['quality_grp']

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)
boostTree = ensemble.AdaBoostClassifier(estimator=classTree, n_estimators=19,
                                        learning_rate=1.0, algorithm='SAMME.R', random_state=20230101)
boostFit = boostTree.fit(x_train, y_train)
boostPredProb = boostFit.predict_proba(x_test)
ytest_pred = numpy.where(boostPredProb[:,1] >= 0.2, 1, 0)
boostAccuracy = metrics.accuracy_score(y_test,ytest_pred)
AUC = metrics.roc_auc_score(y_test, ytest_pred)

print("Area Under Curve on the Testing data using the final converged classification tree :",AUC)
print("Accuracy of the Testing data using the final converged classification tree :",boostAccuracy)

y_testCopy = pandas.DataFrame(y_test)
y_testCopy['P_quality_grp'] = boostPredProb[:,1]
y_testCopy = y_testCopy[y_testCopy['quality_grp'] == 1]
y_testCopy.boxplot(column='P_quality_grp', by='quality_grp', vert = False, figsize=(6,4))
plt.title("boxplot for the predicted probability for quality_grp = 1")
plt.suptitle(" ")
plt.xlabel("Prediction Value of Quality Group")
plt.ylabel("Quality Group")
plt.xlim(-0.1,1.1)
plt.grid(axis="y")
plt.xticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()