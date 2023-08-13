from xgboost import XGBClassifier
import xgboost as xgb
import smart_open
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime
import pandas as pd
import warnings
from io import StringIO
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

s3 = boto3.client('s3')


train = pd.read_parquet((
    '~/Downloads/customer_campaign_table_complete_PROP_MODEL.parquet'))


targets = train.purchased_or_not
predictors = train.drop(['email', 'campaign_name',
       'purchased_or_not','average_time_between_opened_email_days',
       'average_time_between_clicked_email_days',
       'average_time_between_purchase_days', 'time_first_last_opened_email',
       'time_first_last_clicked_email', 'time_first_last_purchase',
       'days_since_last_opened_email', 'days_since_last_clicked_email',
       'days_since_last_purchase', 
       'customer_lifetime_gross_sales', 'customer_lifetime_number_of_units',
       'average_time_between_opened_email_days_difference',
       'average_time_between_clicked_email_days_difference',
       'time_first_last_opened_email_difference',
       'time_first_last_clicked_email_difference',
       'days_since_last_opened_email_difference',
       'days_since_last_clicked_email_difference',
       'days_since_last_purchase_from_campaign_difference'], axis=1)


oversample = SMOTE(sampling_strategy=0.20, random_state=42)
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(
    predictors, targets, stratify=targets, test_size=0.2)


X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train)
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)


from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))




# A parameter grid for XGBoost
params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5],
    'scale_pos_weight': [1,2,3,4],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 400, 500]
}
xgb = XGBClassifier(objective='binary:logistic',
                    silent=True, nthread=1)

folds = 5
param_comb = 6

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb,
                                   scoring='neg_log_loss', n_jobs=3, cv=skf.split(X_train_smote, y_train_smote), verbose=3, random_state=1001)

# Here we go
# timing starts from this point for "start_time" variable
start_time = timer(None)
random_search.fit(X_train_smote, y_train_smote)
timer(start_time)  # timing ends here for "start_time" variable

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' %
      (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('results.csv')



data = pd.read_parquet('~/Downloads/customer_campaign_table_complete_PROP_MODEL.parquet')

train=data
targets = train.purchased_or_not
predictors = train.drop(['email', 'campaign_name',
       'purchased_or_not','average_time_between_opened_email_days',
       'average_time_between_clicked_email_days',
       'average_time_between_purchase_days', 'time_first_last_opened_email',
       'time_first_last_clicked_email', 'time_first_last_purchase',
       'days_since_last_opened_email', 'days_since_last_clicked_email',
       'days_since_last_purchase', 
       'customer_lifetime_gross_sales', 'customer_lifetime_number_of_units',
       'average_time_between_opened_email_days_difference',
       'average_time_between_clicked_email_days_difference',
       'time_first_last_opened_email_difference',
       'time_first_last_clicked_email_difference',
       'days_since_last_opened_email_difference',
       'days_since_last_clicked_email_difference',
       'days_since_last_purchase_from_campaign_difference'], axis=1)
print('1')
oversample = SMOTE(sampling_strategy = 0.20,random_state=42)
scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(predictors, targets, stratify=targets, test_size=0.01)
X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train)

X_train_smote = scaler.fit_transform(X_train_smote)
X_test= scaler.transform(X_test)

print('before fit')
model = XGBClassifier(learning_rate=0.3,
                            max_depth = 4, 
                            n_estimators = 100,
                              scale_pos_weight=1,
                              gamma=1)

model.fit(X_train_smote, y_train_smote)
print('after fit, about to predicts')
ypred = model.predict_proba(X_test)

df = pd.DataFrame([])
i=0
for campaign in data.campaign_name.unique():
   i+=1
   X_test = data[data['campaign_name']==campaign]
   X_test_target = X_test.purchased_or_not
   X_test_predictors = X_test.drop(['email', 'campaign_name',
      'purchased_or_not','average_time_between_opened_email_days',
      'average_time_between_clicked_email_days',
      'average_time_between_purchase_days', 'time_first_last_opened_email',
      'time_first_last_clicked_email', 'time_first_last_purchase',
      'days_since_last_opened_email', 'days_since_last_clicked_email',
      'days_since_last_purchase', 
      'customer_lifetime_gross_sales', 'customer_lifetime_number_of_units',
      'average_time_between_opened_email_days_difference',
      'average_time_between_clicked_email_days_difference',
      'time_first_last_opened_email_difference',
      'time_first_last_clicked_email_difference',
      'days_since_last_opened_email_difference',
      'days_since_last_clicked_email_difference',
      'days_since_last_purchase_from_campaign_difference'], axis=1)
   X_test_predictors= scaler.transform(X_test_predictors)
   ypred = model.predict_proba(X_test_predictors)
   ypred = pd.DataFrame(ypred)
   ypred = ypred.iloc[:,[1]]
   ypred.columns = ['score']
   try1 = pd.concat([ypred.reset_index(), X_test.email.reset_index()], axis=1)
   try1.drop('index', axis=1, inplace=True)
   try1['campaign_name'] = campaign
   df = df.append(try1, ignore_index=True)
   print(campaign)


























