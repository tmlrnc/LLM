from datetime import datetime, timedelta,date
import pandas as pd
%matplotlib inline
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from __future__ import division
from sklearn.cluster import KMeans


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




HOME_DIR = '/Users/thomaslorenc/Sites/flag/forecast/data/'
sales_file_in =  HOME_DIR + 'recs_train.parquet'
tx_data = pd.read_parquet(sales_file_in)
#tx_data = pd.read_csv('data.csv')


tx_data['InvoiceDate'] = pd.to_datetime(tx_data['datetime'])


#tx_data['InvoiceDate'].agg(['min', 'max']) 

tx_6m = tx_data[(tx_data.InvoiceDate < date(2022,8,1)) & (tx_data.InvoiceDate >= date(2019,1,6))].reset_index(drop=True)
tx_next = tx_data[(tx_data.InvoiceDate >= date(2022,8,1)) & (tx_data.InvoiceDate < date(2022,11,1))].reset_index(drop=True)

tx_user = pd.DataFrame(tx_6m['customer_id'].unique())
tx_user.columns = ['customer_id']

