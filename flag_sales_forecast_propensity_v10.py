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

from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.cluster import KMeans

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
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta,date

HOME_DIR = '/home/ec2-user/forecast/'
sales_file_in =  HOME_DIR + 'recs_train.parquet'
train = pd.read_parquet(sales_file_in)
print(train.head(10))


tx_data = pd.read_parquet(sales_file_in)
#tx_data = pd.read_csv('data.csv')


tx_data['InvoiceDate'] = pd.to_datetime(tx_data['datetime'])

#print(tx_data['InvoiceDate'].agg(['min', 'max']))

print(train.columns)


tx_6m = tx_data[(tx_data.InvoiceDate < np.datetime64("2022-08-01")) & (tx_data.InvoiceDate >= np.datetime64("2020-08-01"))].reset_index(drop=True)
tx_next = tx_data[(tx_data.InvoiceDate >= np.datetime64("2022-08-01")) & (tx_data.InvoiceDate < np.datetime64("2022-11-01"))].reset_index(drop=True)
tx_user = pd.DataFrame(tx_6m['customer_id'].unique())
tx_user.columns = ['customer_id']

tx_next_first_purchase = tx_next.groupby('customer_id').InvoiceDate.min().reset_index()
tx_next_first_purchase.columns = ['customer_id','MinPurchaseDate']
print(tx_next_first_purchase.head())

tx_last_purchase = tx_6m.groupby('customer_id').InvoiceDate.max().reset_index()
tx_last_purchase.columns = ['customer_id','MaxPurchaseDate']
tx_purchase_dates = pd.merge(tx_last_purchase,tx_next_first_purchase,on='customer_id',how='left')
tx_purchase_dates['NextPurchaseDay'] = (tx_purchase_dates['MinPurchaseDate'] - tx_purchase_dates['MaxPurchaseDate']).dt.days

print(tx_purchase_dates.head())

tx_user = pd.merge(tx_user, tx_purchase_dates[['customer_id','NextPurchaseDay']],on='customer_id',how='left')
print(tx_user.head())
tx_user = tx_user.fillna(999)

tx_max_purchase = tx_6m.groupby('customer_id').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['customer_id','MaxPurchaseDate']
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
tx_user = pd.merge(tx_user, tx_max_purchase[['customer_id','Recency']], on='customer_id')

print(tx_user.head())
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

#order recency clusters
tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

#print cluster characteristics
tx_user.groupby('RecencyCluster')['Recency'].describe()


#get total purchases for frequency scores
tx_frequency = tx_6m.groupby('customer_id').InvoiceDate.count().reset_index()
tx_frequency.columns = ['customer_id','Frequency']

#add frequency column to tx_user
tx_user = pd.merge(tx_user, tx_frequency, on='customer_id')


#clustering for frequency
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

#order frequency clusters and show the characteristics
tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
tx_user.groupby('FrequencyCluster')['Frequency'].describe()

#create a dataframe with CustomerID and Invoice Date
tx_day_order = tx_6m[['customer_id','InvoiceDate']]
#convert Invoice Datetime to day
tx_day_order['InvoiceDay'] = tx_6m['InvoiceDate'].dt.date
tx_day_order = tx_day_order.sort_values(['customer_id','InvoiceDate'])
#drop duplicates
tx_day_order = tx_day_order.drop_duplicates(subset=['customer_id','InvoiceDay'],keep='first')

#shifting last 3 purchase dates
tx_day_order['PrevInvoiceDate'] = tx_day_order.groupby('customer_id')['InvoiceDay'].shift(1)
tx_day_order['T2InvoiceDate'] = tx_day_order.groupby('customer_id')['InvoiceDay'].shift(2)
tx_day_order['T3InvoiceDate'] = tx_day_order.groupby('customer_id')['InvoiceDay'].shift(3)

tx_day_order['DayDiff'] = (tx_day_order['InvoiceDay'] - tx_day_order['PrevInvoiceDate']).dt.days
tx_day_order['DayDiff2'] = (tx_day_order['InvoiceDay'] - tx_day_order['T2InvoiceDate']).dt.days
tx_day_order['DayDiff3'] = (tx_day_order['InvoiceDay'] - tx_day_order['T3InvoiceDate']).dt.days

tx_day_diff = tx_day_order.groupby('customer_id').agg({'DayDiff': ['mean','std']}).reset_index()
tx_day_diff.columns = ['customer_id', 'DayDiffMean','DayDiffStd']

tx_day_order_last = tx_day_order.drop_duplicates(subset=['customer_id'],keep='last')

tx_day_order_last = tx_day_order_last.dropna()
tx_day_order_last = pd.merge(tx_day_order_last, tx_day_diff, on='customer_id')
tx_user = pd.merge(tx_user, tx_day_order_last[['customer_id','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='customer_id')
#create tx_class as a copy of tx_user before applying get_dummies
tx_class = tx_user.copy()
tx_class = pd.get_dummies(tx_class)

print(tx_user.head())

tx_class['NextPurchaseDayRange'] = 2
tx_class.loc[tx_class.NextPurchaseDay>20,'NextPurchaseDayRange'] = 1
tx_class.loc[tx_class.NextPurchaseDay>50,'NextPurchaseDayRange'] = 0

#train & test split
tx_class = tx_class.drop('NextPurchaseDay',axis=1)
X, y = tx_class.drop('NextPurchaseDayRange',axis=1), tx_class.NextPurchaseDayRange
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
print('Accuracy of XGB classifier on training set: {:.2f}'.format(xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'.format(xgb_model.score(X_test[X_train.columns], y_test)))
y_pred = xgb_model.predict(X_test[X_train.columns])


mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error Value for Light GBM model = {mae}")

prop_score = xgb_model.predict_proba(X_test[X_train.columns])

my_cust_e = train['customer_id'].unique()
my_lol_out_prop = []
my_h2 = ['customer_id', 'propensity_score' ]
my_lol_out_prop.append(my_h2)


k =  0
for ele in prop_score:

    myr = [my_cust_e[k],ele[1]]
    my_lol_out_prop.append(myr)
    k = k + 1


prop_out_data2 = '/home/ec2-user/forecast/SCORES.csv'

with open(prop_out_data2, "w") as f:
    writer = csv.writer(f)
    writer.writerows(my_lol_out_prop)













