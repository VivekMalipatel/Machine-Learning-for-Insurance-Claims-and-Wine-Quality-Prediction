import pandas
import warnings
from itertools import combinations
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

train_counts = train_data [yName].value_counts().sort_index()

print("Training Partition:")
print("Group 0 (no claims):", train_counts[1])
print("Group 1 (1 claim):", train_counts[2])
print("Group 2 (2 claims):", train_counts[3])
print("Group 3 (3 or more claims):", train_counts[4])


test_counts = test_data[yName].value_counts().sort_index()

print("\nTesting Partition:")
print("Group 0 (no claims):", test_counts[1])
print("Group 1 (1 claim):", test_counts[2])
print("Group 2 (2 claims):", test_counts[3])
print("Group 3 (3 or more claims):", test_counts[4])

