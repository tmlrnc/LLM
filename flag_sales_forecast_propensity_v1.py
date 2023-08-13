import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import csv
from xgboost import XGBClassifier
import imblearn
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 


HOME_DIR = '/Users/thomaslorenc/Sites/flag/forecast/data/'
sales_file_in =  HOME_DIR + 'recs_train.parquet'
train_parq = pd.read_parquet(sales_file_in)

print(train_parq.columns)



CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }



train_sales_file_in =  HOME_DIR + 'train.csv'
test_sales_file_in =  HOME_DIR + 'test.csv'

train = pd.read_csv(train_sales_file_in)
test = pd.read_csv(test_sales_file_in)

# DATES FEATURES
def date_features(df):
    # Date Features
    df['date'] = pd.to_datetime(dataset['date'])
    df['year'] = dataset.date.dt.year
    df['month'] = dataset.date.dt.month
    df['day'] = dataset.date.dt.day
    df['dayofyear'] = dataset.date.dt.dayofyear
    df['dayofweek'] = dataset.date.dt.dayofweek
    df['weekofyear'] = dataset.date.dt.weekofyear
    
    # Additionnal Data Features
    df['day^year'] = np.log((np.log(dataset['dayofyear'] + 1)) ** (dataset['year'] - 2000))
    
    # Drop date
    df.drop('date', axis=1, inplace=True)
    
    return df

# Dates Features for Train, Test
train, test = date_features(train), date_features(test)

# Daily Average, Monthly Average for train
train['daily_avg']  = train.groupby(['item','store','dayofweek'])['sales'].transform('mean')
train['monthly_avg'] = train.groupby(['item','store','month'])['sales'].transform('mean')
train = train.dropna()

# Average sales for Day_of_week = d per Item,Store
daymonth_avg = train.groupby(['item','store','dayofweek'])['sales'].mean().reset_index()
# Average sales for Month = m per Item,Store
monthly_avg = train.groupby(['item','store','month'])['sales'].mean().reset_index()


# Merge Test with Daily Avg, Monthly Avg
def merge(df1, df2, col,col_name):
    
    df1 =pd.merge(df1, df2, how='left', on=None, left_on=col, right_on=col,
            left_index=False, right_index=False, sort=True,
             copy=True, indicator=False)
    
    df1 = df1.rename(columns={'sales':col_name})
    return df1

# Add Daily_avg and Monthly_avg features to test 
test = merge(test, daily_avg,['item','store','dayofweek'],'daily_avg')
test = merge(test, monthly_avg,['item','store','month'],'monthly_avg')

# Sales Rolling mean sequence per item 
rolling_10 = train.groupby(['item'])['sales'].rolling(10).mean().reset_index().drop('level_1', axis=1)
train['rolling_mean'] = rolling_10['sales'] 

# 90 last days of training rolling mean sequence added to test data
rolling_last90 = train.groupby(['item','store'])['rolling_mean'].tail(90).copy()
test['rolling_mean'] = rolling_last90.reset_index().drop('index', axis=1)

# Shifting rolling mean 3 months
train['rolling_mean'] = train.groupby(['item'])['rolling_mean'].shift(90) # Create a feature with rolling mean of day - 90
train.head()


# Clean features highly correlated to each others
for df in [train, test]:
    df.drop(['dayofyear', 
                  'weekofyear',
                  'daily_avg',
                  'day',
                  'month',
                  'item',
                  'store',],
                 axis=1, 
                 inplace=True)
    
# Features Scaling (except sales)
sales_series, id_series = train['sales'], test['id']
# Features Scaling
train = (train - train.mean()) / train.std()
test = (test - test.mean()) / test.std()
# Retrieve actual Sales values and ID
train['sales'] = sales_series
test['id'] = id_series

# Training Data
X_train = train.drop('sales', axis=1).dropna()
y_train = train['sales']
# Test Data
test.sort_values(by=['id'], inplace=True)
X_test = test.drop('id', axis=1)
#df = train
df_train = train.copy()

# Train Test Split
X_train , X_test ,y_train, y_test = train_test_split(df_train.drop('sales',axis=1),df_train.pop('sales'), random_state=123, test_size=0.2)

# XGB Model
matrix_train = xgb.DMatrix(X_train, label = y_train)
matrix_test = xgb.DMatrix(X_test, label = y_test)

# Run XGB 
model = xgb.train(params={'objective':'reg:linear','eval_metric':'mae'}
                ,dtrain = matrix_train, num_boost_round = 500, 
                early_stopping_rounds = 20, evals = [(matrix_test,'test')],)



exit()

# Sales rolling mean 
def roll_mean(df_sales, n_days):
    df_roll = df_sales.rolling(n_days).mean().round(0)
 
    return df_roll.dropna()

# Sum of the forecast sales of the next n days
def forecastxgb_sum_n(df_fcst, n_window, rolling_ndays):
    df_ft3n = df_fcst.rolling(window=n_window).sum().shift(-2)
    df_ft3n = df_ft3n.iloc[rolling_ndays-1:]

    return df_ft3n

# p * day n sales applied for the sales of the next n days
def forecastrm_sum_n(df_roll, n_window):
    return df_roll * n_window

# Calculate Sum of Actual sales: Day n, Day n+1, Day n+2, Day n + p
def actual_sum_p(df_sales, days_p):
    df_act_p = df_sales.rolling(window = days_p).sum().shift(-(days_p-1))
    return df_act_p

# Error Forecast Calculation
def error_calc(df_sales, df_fcst, df_roll, rolling_ndays, frcst_n_days):
    
    # Rolling means 
    df_roll = roll_mean(df_sales, rolling_ndays)

    # Sum of the forecast sales of the next n days
    df_ft_n = forecastxgb_sum_n(df_fcst, frcst_n_days, rolling_ndays)

    # Sum of the rolling mean sales of the next n days
    df_rm_n = forecastrm_sum_n(df_roll, frcst_n_days)

    # Calculate delta forecast
    delta_ft, delta_rm, delta_ft_n, df_sales_n, delta_rm_n, delta_rm_max, delta_ft_max = delta_frcst(df_sales, df_ft_n, df_rm_n, 
                                                                                         frcst_n_days, rolling_ndays)
    
    return delta_ft_n, df_sales_n, delta_rm_n, delta_rm_max, delta_ft_max, df_ft_n, df_rm_n

exit()


my_test_camp = 'SS2022-ZIPLINELAUNCH'
train = train.loc[train['campaign_name']!=my_test_camp]
targets = train.purchased_or_not
predictors = train.drop(['purchased_or_not', 'email_hash','campaign_name','campaign_name_label'], axis=1)
predictors.dropna()
targets.dropna()

oversample = SMOTE(sampling_strategy = 0.20,random_state=42)
from sklearn.metrics import mean_squared_error

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(predictors, targets, stratify=targets, test_size=0.20)
X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train)

X_train_smote = scaler.fit_transform(X_train_smote)
X_test= scaler.transform(X_test)


model = XGBClassifier(learning_rate=0.2,
                            max_depth = 3, 
                            n_estimators = 400,
                              scale_pos_weight=20)

model.fit(X_train_smote, y_train_smote)
model.save_model("xbg_model_SS2022-ZIPLINELAUNCH.json")


ypred = model.predict(X_test)
mse = mean_squared_error(y_test, ypred)
print("RMSE XGB: %.2f" % (mse**(1/2.0)))

print('XGB confusionmatrix ')
print(classification_report(y_test, ypred))

print('*********************************')
print("Accuracy XGBClassifier on training set: {:.3f}".format(model.score(X_train_smote, y_train_smote)))

print("Accuracy on validation set: {:.3f}".format(model.score(X_test, y_test)))

