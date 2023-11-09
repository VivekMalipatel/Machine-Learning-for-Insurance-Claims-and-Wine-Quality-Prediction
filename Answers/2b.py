import numpy
import pandas
import sklearn.tree as tree

trainData = pandas.read_csv('WineQuality_Train.csv', delimiter=',')
nObs = trainData.shape[0]

x_train = trainData[['alcohol', 'citric_acid', 'free_sulfur_dioxide', 'residual_sugar', 'sulphates']]
y_train = trainData['quality_grp']

w_train = numpy.full(nObs, 1.0)
accuracy = numpy.zeros(100)
ensemblePredProb = numpy.zeros((nObs, 2))

result = []

classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=20230101)
for iter in range(100):
    treeFit = classTree.fit(x_train, y_train, w_train)
    treePredProb = classTree.predict_proba(x_train)
    y_predClass = numpy.where(treePredProb[:,1] >= 0.2, 1, 0)
    accuracy[iter] = numpy.sum(numpy.where(y_train == y_predClass, w_train, 0.0)) / numpy.sum(w_train)
    result.append([iter,numpy.sum(w_train),accuracy[iter]])
    
    if (abs(1.0 - accuracy[iter]) < 0.0000001):
        print("Number of Iterations performed to achieve convergence : ",iter+1)
        break
    
    # Update the weights
    eventError = numpy.where(y_train == 1, (1 - treePredProb[:,1]), (treePredProb[:,1]))
    w_train = numpy.where(y_predClass != y_train , 2+numpy.abs(eventError), numpy.abs(eventError))

result_df = pandas.DataFrame(result, columns = ['Iteration','Sum of Weights','Weighted Accuracy'])
result_df.to_csv('2bOutput.csv')    