# Machine-Learning-for-Insurance-Claims-and-Wine-Quality-Prediction
CS 484: Introduction to Machine Learning Assignment

This assignment for CS 484: Introduction to Machine Learning during the Spring Semester 2023 involves predictive modeling for insurance claims and wine quality assessment using various machine learning techniques.

## Question 1: Predictive Modeling for Insurance Claims (50 points)

The `Homeowner_Claim_History.xlsx` file contains claim history data for 27,513 homeowner policies. The task is to predict the number of claims using the provided features, categorizing the policies into four groups based on the number of claims.

### Data Description

- **policy**: Policy Identifier
- **exposure**: Duration a Policy is Exposed to Risk (Portion of a Year)
- **num_claims**: Number of Claims in a Year
- **amt_claims**: Total Claim Amount in a Year
- **f_primary_age_tier**: Age Tier of Primary Insured (< 21, 21 - 27, 28 - 37, 38 - 60, > 60)
- **f_primary_gender**: Gender of Primary Insured (Female, Male)
- **f_marital**: Marital Status of Primary Insured (Not Married, Married, Un-Married)
- **f_residence_location**: Location of Residence Property (Urban, Suburban, Rural)
- **f_fire_alarm_type**: Fire Alarm Type (None, Standalone, Alarm Service)
- **f_mile_fire_station**: Distance to Nearest Fire Station (< 1 mile, 1 - 5 miles, 6 - 10 miles, > 10 miles)
- **f_aoi_tier**: Amount of Insurance Tier (< 100K, 100K - 350K, 351K - 600K, 601K - 1M, > 1M)

### Tasks

(a) Determine the number of policies in each group for both Training and Testing partitions.
(b) Identify the model with the lowest AIC value on the Training partition.
(c) Find the model with the lowest BIC value on the Training partition.
(d) Determine the model with the highest Accuracy on the Testing partition.
(e) Ascertain the model with the lowest Root Average Squared Error on the Testing partition.

## Question 2: Wine Quality Prediction Using Adaptive Boosting (50 points)

We will analyze datasets from the UCI Machine Learning Repository to train and test predictive models for wine quality.

### Dataset Information

- **Target variable**: `quality_grp` (0 or 1)
- **Input features**: `alcohol`, `citric_acid`, `free_sulfur_dioxide`, `residual_sugar`, `sulphates`

### Model Specifications

- Entropy as the Splitting Criterion
- Maximum tree depth of five
- Random state initialization at 20230101
- Maximum of 100 Boosting iterations
- Convergence criteria and weighting adjustments based on classification accuracy and errors

### Evaluation Criteria

(a) Calculate the Misclassification Rate on Training data at Iteration 0.
(b) Document the number of iterations to achieve convergence with iteration history.
(c) Determine the Area Under Curve on Testing data with the final converged tree.
(d) Calculate the Accuracy on Testing data with the final converged tree.
(e) Create a grouped boxplot for the predicted probability of `quality_grp = 1` on Testing data, grouped by observed categories.
