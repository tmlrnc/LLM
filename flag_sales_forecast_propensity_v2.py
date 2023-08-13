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



ypred = model.predict(X_test)
mse = mean_squared_error(y_test, ypred)
print("RMSE XGB: %.2f" % (mse**(1/2.0)))

print('XGB confusionmatrix ')
print(classification_report(y_test, ypred))

print('*********************************')
print("Accuracy XGBClassifier on training set: {:.3f}".format(model.score(X_train_smote, y_train_smote)))

print("Accuracy on validation set: {:.3f}".format(model.score(X_test, y_test)))


model_xgb_2 = xgb.XGBClassifier()
model_xgb_2.load_model("xbg_model_SS2022-LAUNCH.json")

sales_file_in = '/Users/thomaslorenc/Sites/flag/prop/data/active_APL_tom-propensity_customer_campaign_table_ENGAGED_v15.parquet'

df_sales_file_in = pd.read_parquet(sales_file_in)
my_test_camp = 'SS2022-LAUNCH'
train = df_sales_file_in[df_sales_file_in['campaign_name'] == my_test_camp] 


targets = train.purchased_or_not
X_test_prop = train.drop([ 'customer_id','ordered_item_quantity'], axis=1)

my_cust_e = train['customer_id'].unique()


scaler = MinMaxScaler()



X_test_prop = scaler.fit_transform(X_test_prop)

prop_score = model_xgb_2.predict_proba(X_test_prop)

my_lol_out_prop = []
my_h2 = ['email_hash', 'propensity_score' ]
my_lol_out_prop.append(my_h2)


k =  0
for ele in prop_score:

    myr = [my_cust_e[k],ele[1]]
    my_lol_out_prop.append(myr)
    k = k + 1



prop_out_data2 = '/Users/thomaslorenc/Sites/flag/prop/data/SS2022-LAUNCH_SCORES.csv'

with open(prop_out_data2, "w") as f:
    writer = csv.writer(f)
    writer.writerows(my_lol_out_prop)

exit()

import pandas as pd
import datetime as dt


train_sales_file_in =  HOME_DIR + 'train.csv'
train = pd.read_csv(train_sales_file_in)

test_sales_file_in =  HOME_DIR + 'test.csv'

test = pd.read_csv(test_sales_file_in)


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


print('train')
check_df(train)
print('test')

check_df(test)









train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
train["year_month"] = train["date"].dt.to_period('M')
train["year_month"] = train["year_month"].astype(str)


df = train.groupby(["year_month", "store", "item"]).agg({"sales":"sum"})
df = df.reset_index()
df.tail()
df.info()

df.groupby("store").agg({"sales":"sum"}).sort_values (by="sales", ascending=False).head()
df.groupby("store").agg({"sales":"sum"}).sort_values (by="sales").head()

df.groupby("item").agg({"sales":"sum"}).sort_values(by="sales", ascending=False).head()
df.groupby("item").agg({"sales":"sum"}).sort_values (by="sales").head()


df.groupby("store").agg({"item":"count"}) #no missing values
df.groupby("store").agg({"sales":"sum"}).describe().T #no outlier for total 



#LIGHT GBM 
#Apart from other forecasting methods, in Light GBM we will derive new features for our model
#First we construct a function to derive new date features

df["month"] = df["year_month"].str[5:]
df["year"] = df["year_month"].str[:4].astype(int)

#We will add lag features for sales variable (3 months, 6 months, 12 months)
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag))
    return dataframe
df = lag_features(df, [3, 6, 12])
#We will add rolling mean features for sales variable (3 months, 6 months, 12 months, 15 months)
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=2, win_type="triang").mean())
    return dataframe
df = roll_mean_features(df, [3, 6, 12, 15])

#Finally, we will add exponentially weighted mean features (3 months, 6 months, 12 months, 15 months)
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [3, 6, 9, 12, 15]
df = ewm_features(df, alphas, lags)



df = pd.get_dummies(df, columns=['store', 'item', 'month']) #One-hot encoding
df['sales'] = np.log1p(df["sales"].values) #taking the logarithm of dependent variable to inrease computational efficiency

train_lgbm = df.loc[df["year"].astype(int)<2017]
val_lgbm = df.loc[df["year"].astype(int)==2017]

cols = [col for col in train_lgbm.columns if col not in ["id", "sales", "year_month", "year"]]

X_train = train_lgbm[cols]
Y_train = train_lgbm["sales"]

X_val = val_lgbm[cols]
Y_val = val_lgbm["sales"]

import lightgbm as lgb
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],                  
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

mae = mean_absolute_error(Y_val, y_pred_val)
print(f"Mean Absolute Error Value for Light GBM model = {mae}")

exit()



