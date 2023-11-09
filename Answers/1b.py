import numpy
import pandas
import sys
import time
from itertools import combinations
import Utility
import warnings
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

   for pred in modelTerm:
      if (pred in catName):
         X_train = X_train.join(pandas.get_dummies(trainData[pred].astype('category')))

   resultList = Utility.MNLogisticModel (X_train, y_train, maxIter = maxIter, tolSweep = tolS)

   modelLLK = resultList[1]
   modelDF = resultList[2]
   del resultList

   AIC = 2.0 * modelDF - 2.0 * modelLLK
   BIC = modelDF * numpy.log(n_sample) - 2.0 * modelLLK
   allCombResult.append([r, modelTerm, len(modelTerm), modelLLK, modelDF, AIC, BIC, n_sample])

endTime = time.time()

allCombResult = pandas.DataFrame(allCombResult, columns = ['Step', 'Model Term', 'Number of Terms', \
                'Log-Likelihood', 'Model Degree of Freedom', 'Akaike Information Criterion', \
                'Bayesian Information Criterion', 'Sample Size'])
elapsedTime = endTime - startTime

min_aic = allCombResult.loc[allCombResult['Akaike Information Criterion'].idxmin()]
print("Lowest AIC value on the Training partition",min_aic['Akaike Information Criterion'])
print("model producing the Lowest AIC value on the Training partition",min_aic['Model Term'])

