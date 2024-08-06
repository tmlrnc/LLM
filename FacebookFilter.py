# Databricks notebook source
# MAGIC %md
# MAGIC # Facebook Filter
# MAGIC
# MAGIC The purpose of this notebook is to read in the latest facebook posts and detect whether or not those posts need to be further processed for potential closure information.
# MAGIC
# MAGIC ## Inputs
# MAGIC * schoolclosure.facebook TABLE
# MAGIC * inclusion_terms.txt - file of inclusion terms for filter
# MAGIC * exclusion_terms.txt - file of exclusion terms for filter
# MAGIC
# MAGIC ## Processes
# MAGIC
# MAGIC ## Outputs
# MAGIC * schoolclosure.facebook `processed_at` field update
# MAGIC * schoolclosure.facebook_filtered TABLE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install tqdm nltk

# COMMAND ----------

# Standard Imports
import re
import os
import datetime as dt
import json
import pyspark.sql.functions as f

# Third Party Imports
import pandas as pd
import requests
import nltk
from nltk import word_tokenize
from tqdm import tqdm

# COMMAND ----------

# Setup tqdm
tqdm.pandas()

# Ingest punkt
nltk.download('punkt')

# COMMAND ----------

# Ingest secrets
storage_account_name = "schoolclosurestorage"
storage_account_key = dbutils.secrets.get(scope = "schoolclosure", key = "storageaccountkey")
container = "schoolclosure"

# Set variable with key for blob storage
spark.conf.set(f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net", storage_account_key)

# COMMAND ----------

dbutils.fs.ls(f"wasbs://{container}@{storage_account_name}.blob.core.windows.net")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest

# COMMAND ----------

# DBTITLE 1,Ingest Inclusion/Exclusion Terms
inclusion_terms = spark.read.format("text").load('wasbs://schoolclosure@schoolclosurestorage.blob.core.windows.net/inclusion_terms.txt', header=False)
exclusion_terms = spark.read.format("text").load('wasbs://schoolclosure@schoolclosurestorage.blob.core.windows.net/exclusion_terms.txt', header=False)

# Convert dataframes to lists of strings
inclusion_terms_list = [str(row.value) for row in inclusion_terms.collect()]
exclusion_terms_list = [str(row.value) for row in exclusion_terms.collect()]

# Convert exclusion terms to regex
exclude_regex_terms = [f"\\b{x}\\b" for x in exclusion_terms_list]
exclude_regex = "|".join(exclude_regex_terms)

print("INCLUSION TERMS: ", inclusion_terms_list[0:5], "...")
print("EXCLUSION TERMS: ", exclusion_terms_list[0:5], "...")
print("EXCLUSION REGEX: ", exclude_regex[0:50], "...")

# COMMAND ----------

# DBTITLE 1,Ingest Unprocessed Posts
# Ingest from hive metastore
unprocessed_df = spark.sql('SELECT * FROM schoolclosure_adl.facebook_posts WHERE processed_date IS NULL;')
unprocessed_rows = unprocessed_df.cache().count()
print(f"INGESTED {unprocessed_rows} UNPROCESSED POSTS")

# If no records to process, exit
if unprocessed_rows==0:
    print("NO RECORDS TO PROCESS, EXITING")
    dbutils.notebook.exit("NO RECORDS TO PROCESS")

# COMMAND ----------

# Convert to pandas dataframe
unprocessed_df_slim = unprocessed_df.select('created_time', 'id', 'message',
                                            'permalink_url', 'full_text', 'facebook_account_id')
unprocessed_pd = unprocessed_df_slim.toPandas()

# Show data
display(unprocessed_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing

# COMMAND ----------

def preprocess(text):
    """Pre-processes the text, splits into tokens that are lower-cased, filtered."""

    # Handle nulls
    if not text:
        return None

    # Replace hyphens with spaces to ensure compound terms are included
    text = text.replace('-', ' ')

    # Retain lower-case alphabetic words: alpha_only
    tokens = (token.lower() for token in word_tokenize(text)
                            if token.isalpha())

    return " ".join(tokens)

def filter_tweet_conditional(preprocessed: str, filter_list: list, exclude_regex: str) -> bool:
    """
    Filters tweets based on inclusion and exclusion criteria
    
    Parameters
    ----------
    preprocessed
        text of a given tweet, already preprocessed
    filter_list:
        list of strings to use for filtering
    exclue_regex:
        regex string of terms to trigger exclusion
        
    Returns
    -------
    bool
        whether or not a tweet should be considered further. If false, tweet is discarded as not closure-related.
    """

    # Exclude tweets which contain substring from exclude list
    if re.search(exclude_regex, preprocessed):
        return False
    else:
        pass

    # Include tweets which contain substring from include list
    for i in filter_list:
        if i in preprocessed:
            return True
        else:
            continue

    return False

def filter_tweet_conditional(preprocessed: str, filter_list: list, exclude_regex: str) -> bool:
    """
    Filters tweets based on inclusion and exclusion criteria
    
    Parameters
    ----------
    preprocessed
        text of a given tweet, already preprocessed
    filter_list:
        list of strings to use for filtering
    exclue_regex:
        regex string of terms to trigger exclusion
        
    Returns
    -------
    bool
        whether or not a tweet should be considered further. If false, tweet is discarded as not closure-related.
    """

    if not preprocessed:
        return False

    # Exclude tweets which contain substring from exclude list
    if re.search(exclude_regex, preprocessed):
        return False
    else:
        pass

    # Include tweets which contain substring from include list
    for i in filter_list:
        if i in preprocessed:
            return True
        else:
            continue

    return False

def filter_post_df(df: pd.DataFrame, inclusion_terms_list: list, exclude_regex: str) -> pd.DataFrame:
    """
    Preprocess text, compile terms to filter, filter text in dataframe.
    
    Parameters
    ----------
    df
        pd.DataFrame with tweets in the 'text' column
    inclusion_terms_list
        list of strings with inclusion terms
    exclude_regex
        string of regex pattern
        
    Returns
    -------
    pd.DataFrame
        tweets along with a 'filtered' column with boolean values, indicating if a tweet should be further explored
    """

    # Preprocess text
    df['preprocessed'] = df['text'].apply(preprocess)

    # Apply filtering
    df['filtered'] = df.progress_apply(lambda x: filter_tweet_conditional(x['preprocessed'], inclusion_terms, exclude_regex), axis=1)

    return df

# COMMAND ----------

# Preprocess text
unprocessed_pd['preprocessed'] = unprocessed_pd['full_text'].progress_apply(preprocess)

# COMMAND ----------

# Apply filtering
unprocessed_pd['filtered'] = unprocessed_pd.progress_apply(lambda x: filter_tweet_conditional(x['preprocessed'], inclusion_terms_list, exclude_regex), axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply filtering

# COMMAND ----------

# Filter to only tweets which are relevant
filtered_df = unprocessed_pd[unprocessed_pd['filtered']==True]
print(f"FILTERED DOWN TO {len(filtered_df)} POSTS")

# COMMAND ----------

filtered_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export

# COMMAND ----------

# Convert to spark table
filtered = spark.createDataFrame(filtered_df)

# Join with original df to get additional data
join_df = unprocessed_df.select('id', 'attachments', 'attachments_text', 'attachment_img', 'attachment_targets') \
                      .withColumnRenamed('id', 'join_id')
filtered = filtered.join(join_df, filtered.id==join_df.join_id) \
                   .drop('join_id')

# COMMAND ----------

filtered.display()

# COMMAND ----------

filtered.columns

# COMMAND ----------

# DBTITLE 1,Format df for write
# Add columns as necessary
filtered = filtered.withColumn('filtered_at', f.current_timestamp()) \
                   .withColumn('analyzed_at', f.lit(None).cast('timestamp'))

# Select columns in order
filtered = filtered.select('id', 'created_time', 'filtered_at', 'analyzed_at',
                           'facebook_account_id', 'permalink_url',
                           'message', 'full_text', 'attachments', 'attachments_text',
                           'attachment_img', 'attachment_targets')

filtered.display()

# COMMAND ----------

# DBTITLE 1,Write filtered posts to facebook_filtered table
filtered.write.saveAsTable('schoolclosure_adl.facebook_filtered', mode='append')

# COMMAND ----------

# DBTITLE 1,Update facebook table with filtered_date
spark.sql('''UPDATE schoolclosure_adl.facebook_posts
                SET processed_date = CURRENT_TIMESTAMP()
              WHERE processed_date IS NULL;
          ''')

# COMMAND ----------

# Exit notebook with success message
dbutils.notebook.exit('SUCCESS')

# COMMAND ----------


