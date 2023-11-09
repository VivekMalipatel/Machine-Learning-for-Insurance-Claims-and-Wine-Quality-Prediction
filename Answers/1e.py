import numpy
import pandas
import sys
import time
from itertools import combinations
import Utility
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

def assign_claims_group(num_claims):
    if num_claims == 0:
        return 1
    elif num_claims == 1:
        return 2
    elif num_claims == 2:
        return 3
    else:
        return 4

inputData = pandas.read_excel('Homeowner_Claim_History.xlsx',sheet_name = 'HOCLAIMDATA')

catName = ['f_primary_age_tier', 'f_primary_gender', 'f_marital', 'f_residence_location', 'f_fire_alarm_type', 'f_mile_fire_station', 'f_aoi_tier']
nPredictor = len(catName)
yName = 'group'

inputData[yName] = inputData["num_claims"].apply(assign_claims_group)

data = inputData[['policy'] +catName + [yName]].dropna().reset_index(drop = True)
data.set_index('policy', inplace=True)

train_data = data[data.index.str[0].isin(['A', 'G', 'Z'])]
test_data = data[~data.index.str[0].isin(['A', 'G', 'Z'])]

allCombResult = []
allFeature = catName

allComb = []
for r in range(nPredictor+1):
   allComb = allComb + list(combinations(allFeature, r))

startTime = time.time()
maxIter = 20
tolS = 1e-7

nComb = len(allComb)
for r in range(nComb):
   modelTerm = list(allComb[r])
   trainData = train_data[[yName] + modelTerm].dropna()
   n_sample = trainData.shape[0]

   X_train = trainData[[yName]].copy()
   X_train.insert(0, 'Intercept', 1.0)
   X_train.drop(columns = [yName], inplace = True)

   y_train = trainData[yName].copy()

   testData = test_data[[yName] + modelTerm].dropna()

   X_test = testData[[yName]].copy()
   X_test.insert(0, 'Intercept', 1.0)
   X_test.drop(columns = [yName], inplace = True)

   y_test = testData[yName].copy()

   for pred in modelTerm:
      if (pred in catName):
         X_train = X_train.join(pandas.get_dummies(trainData[pred].astype('category')))
         X_test = X_test.join(pandas.get_dummies(testData[pred].astype('category')))

   n_param = X_test.shape[1]

   # Identify the aliased parameters
   XtX = X_test.transpose().dot(X_test)
   origDiag = numpy.diag(XtX)
   XtXGinv, aliasParam, nonAliasParam = Utility.SWEEPOperator (n_param, XtX, origDiag, sweepCol = range(n_param), tol = tolS)

   # Train a multinominal logistic model
   X_reduce = X_test.iloc[:, list(nonAliasParam)] 

   resultList = Utility.MNLogisticModel (X_train, y_train, maxIter = maxIter, tolSweep = tolS)
   ytest_pred = resultList[0].predict(X_reduce)
   pred_labels = ytest_pred.idxmax(axis=1)
   accuracy = accuracy_score(y_test,pred_labels)
   mse = mean_squared_error(y_test, pred_labels)
   rmse = numpy.sqrt(mse)

   modelLLK = resultList[1]
   modelDF = resultList[2]
   del resultList

   AIC = 2.0 * modelDF - 2.0 * modelLLK
   BIC = modelDF * numpy.log(n_sample) - 2.0 * modelLLK
   allCombResult.append([r, modelTerm, len(modelTerm), modelLLK, modelDF, AIC, BIC, n_sample,accuracy,rmse])

endTime = time.time()

allCombResult = pandas.DataFrame(allCombResult, columns = ['Step', 'Model Term', 'Number of Terms', \
                'Log-Likelihood', 'Model Degree of Freedom', 'Akaike Information Criterion', \
                'Bayesian Information Criterion', 'Sample Size','Accuracy of TestData','RMSE of TestData'])
elapsedTime = endTime - startTime

allCombResult.to_csv('1output.csv')
min_testrmse = allCombResult.loc[allCombResult['RMSE of TestData'].idxmin()]
print("lowest Root Average Squared Error on the Testing partition",min_testrmse['RMSE of TestData'])
print("model producing the lowest Root Average Squared Error on the Testing partition",min_testrmse['Model Term'])