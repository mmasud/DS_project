# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import time, datetime
from datetime import datetime, date

import logging

# For Modeling
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import joblib 

# Create and Configure logger
LOG_FORMAT = '%(levelname)s %(asctime)s - %(message)s'
logging.basicConfig(filename='C:/Users/masud.pervez/Documents/SEB_project/project.log',
                    format = LOG_FORMAT,
                    level= logging.DEBUG, 
                    filemode= 'w') # 'w' will overwritten the file

logger = logging.getLogger() # Create a logger obj using getLogger()

########################################################
# path to data

PATH= "C:\\Users\\masud.pervez\\Documents\\SEB_project\\seb_project\\data"

########################################################

def calculateAge(dob):
    '''This function will calculate age giving a string/int dob.'''
    today= date.today() 
    dob= pd.to_datetime(dob, format= "%Y%m%d").date()
    #dob = datetime.strptime(str(dob), "%Y%m%d").date() # convert int to str first
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

# Load data
logger.info("Loading datasets.........")
customer_df = pd.read_csv(f'{PATH}//Customer.csv', sep=',')
district_df = pd.read_csv(f'{PATH}//District.csv', sep=',')
transaction_df = pd.read_csv(f'{PATH}//Transaction.csv', sep=',')
print(customer_df.head())

########################################################
# Data Pre-processing

age= customer_df['BIRTH_DT'].apply(calculateAge)
customer_df.insert(4, 'Age', age, True)
customer_df.head()

# # Create Timestamp object
customer_df['BIRTH_DT'] = pd.to_datetime(customer_df.BIRTH_DT, format= "%Y%m%d")
transaction_df['DATE'] = pd.to_datetime(transaction_df.DATE, format= "%Y%m%d")

# Aggregate transaction dataset for merging

transaction_aggdf= transaction_df.groupby('ACCOUNT_ID').agg( 
                                                max_date=('DATE', max),
                                                min_date=('DATE', min),
                                                unique_dates= ('DATE' , "nunique"),
                                                num_days=(
                                                    "DATE", 
                                                    lambda x: (max(x) - min(x)).days),
                                                trans_amount= ('AMOUNT', 'mean'),
                                                avg_balance= ('BALANCE', 'mean'),
                                                type_most= ('TYPE', lambda x: x.value_counts().index[0]),
                                                operation_most= ('OPERATION', lambda x: x.value_counts().index[0])
                                                #type_most= ('TYPE', pd.Series.mode),
                                                #operation_most= ('OPERATION', pd.Series.mode)
                                                ). reset_index()

# merge all the datasets for model creation
logger.info("Creating model dataset.........")
model_df = (pd.merge(customer_df, district_df, how = 'left', on = "DISTRICT_ID")\
            .merge(transaction_aggdf , how ='left', on = 'ACCOUNT_ID')
           )

# Convert gender from F, M to 0,1
model_df['GENDER'].replace({'F':0, 'M':1}, inplace = True)
model_df['DISTRICT_ID']= model_df['DISTRICT_ID'].astype('str')
model_df['UNEMP_95'] = model_df['UNEMP_95'].replace('?', np.nan).astype(float)
model_df['CRIME_95']= model_df['CRIME_95'].replace('?', np.nan).astype(float) 
# replacing missing values
#model_df['UNEMP_95'].fillna(np.mean(model_df['UNEMP_95']),inplace=True )
#model_df['CRIME_95'].fillna(np.mean(model_df['CRIME_95']),inplace=True )


# Train-Test split

train= model_df.loc[model_df.SET_SPLIT== 'TRAIN']
test= model_df.loc[model_df.SET_SPLIT== 'TEST']


# select variables to use for modelling
target= 'LOAN'
IDcols = ['CLIENT_ID', 'ACCOUNT_ID', 'DISTRICT_ID']
deleted_cols= ['BIRTH_DT','SET_SPLIT', 'max_date', 'min_date']

predictors = [x for x in train.columns if x not in target and  x not in IDcols and  x not in deleted_cols]

# drop "LOAN" and assign it to target variable
X = train[predictors]
y = train[target]

X['operation_most'] = np.where((X.operation_most =='COLLECTION_FROM_OTHER_BANK') | (X.operation_most== 'CREDIT_IN_CASH'),
                              'OTHERS', X.operation_most )

# adding dummies to the dataset
X = pd.get_dummies(X)


# Model Development and Evaluation

# split the data into train and cross validation set
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=123,stratify= y)

# take a look at the dimension of the data
print(x_train.shape, x_cv.shape, y_train.shape, y_cv.shape)

# # Model Development and Evaluation

# Set up a pipeline with a feature selection preprocessor that
# selects the top 2 features to use.
# The pipeline then uses a RandomForestClassifier to train the model.

pipeline = Pipeline([
      ('imputer', SimpleImputer()),
      ('feature_selection', SelectKBest(chi2, k=10)), # Select Best 10 feature according to chi2;
      ('rf', RandomForestClassifier(n_estimators=200))
    ])

pipeline.fit(x_train, y_train)

# make prediction
pred_cv = pipeline.predict(x_cv)

# calculate accuracy score
print(accuracy_score(y_cv, pred_cv))

# import confusion_matrix
from sklearn.metrics import confusion_matrix
 
cm = confusion_matrix(y_cv, pred_cv)
print(cm)

# f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')

# Fit model on full train dataset

pipeline.fit(X , y)

pred_train= pipeline.predict(X)
print(accuracy_score(y, pred_train))

# Make prediction on test dataset
X_test= test[predictors]
y_test= test[target]
X_test['operation_most'] = np.where((X_test.operation_most =='COLLECTION_FROM_OTHER_BANK') | (X_test.operation_most== 'CREDIT_IN_CASH'),
                              'OTHERS', X_test.operation_most )

# adding dummies to the dataset
X_test = pd.get_dummies(X_test)
pred_test= pipeline.predict(X_test)
print(accuracy_score(y_test, pred_test))

print("Model performance on train data\n", classification_report(y, pred_train))
print("Model performance on test data\n", classification_report(y_test, pred_test))

# Export the classifier to a file
joblib.dump(pipeline, 'model_rf.joblib')

print("Finish successfully!........")







