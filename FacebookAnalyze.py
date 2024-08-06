# Databricks notebook source
# MAGIC %md
# MAGIC # Facebook Analyze
# MAGIC
# MAGIC The purpose of this notebook is to read in newly filtered facebook posts and detect whether or not they are related to unexpected school closures using a pre-trained fine-tuned roBERTa model.
# MAGIC
# MAGIC ## Inputs
# MAGIC * schoolclosure.facebook_filtered TABLE
# MAGIC * pre-trained model from dbfs:/schoolclosure_model_store/
# MAGIC
# MAGIC ## Processes
# MAGIC
# MAGIC ## Outputs
# MAGIC * schoolclosure.facebook_filtered `analyzed_at` field update
# MAGIC * schoolclosure.facebook_closure TABLE

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install nltk==3.8.1 tqdm==4.66.2 torch==2.2.0 transformers==4.37.2

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
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

# COMMAND ----------

# Setup tqdm
tqdm.pandas()

# COMMAND ----------

# Load pytorch model for inference
torch_model_dir = '/dbfs/schoolclosure_model_store'
print("LOADING PYTORCH MODEL FROM: ", torch_model_dir)
torch_tokenizer = AutoTokenizer.from_pretrained(torch_model_dir, local_files_only=True)
torch_model = AutoModelForSequenceClassification.from_pretrained(torch_model_dir)

# Set model into eval mode
torch_model_eval = torch_model.eval()

# Build analyzer pipeline
analyzer = pipeline("text-classification", model=torch_model_eval, tokenizer=torch_tokenizer, truncation=True)

# COMMAND ----------

# Run pytorch model for inference
text = '''***All Schools - Virtual Learning - Friday October 13, 2023***Due to ongoing cleanup and restoration of the middle high school due to a roof drain failure, our schools will shift to virtual learning tomorrow Friday October 13, 2023. This shift will allow for cleanup and assessment crews to continue work within the middle high school building and focus on mitigation to resume normal school operations on Monday October 16, 2023. Students are expected to login into Google Clas… See more'''
result = analyzer(text)
result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest

# COMMAND ----------

# DBTITLE 1,Ingest Unanalyzed Posts
# Ingest from hive metastore
unanalyzed_df = spark.sql('SELECT * FROM schoolclosure_adl.facebook_filtered WHERE analyzed_at IS NULL;')
unanalyzed_rows = unanalyzed_df.cache().count()
print(f"INGESTED {unanalyzed_rows} UNANALYZED POSTS")

# If no records to process, exit
if unanalyzed_rows==0:
    print("NO RECORDS TO PROCESS, EXITING")
    dbutils.notebook.exit("NO RECORDS TO PROCESS")

# COMMAND ----------

# Convert to pandas dataframe
unanalyzed_df_slim = unanalyzed_df.select('created_time', 'id', 'message',
                                            'permalink_url', 'full_text', 'facebook_account_id')
unanalyzed_pd = unanalyzed_df_slim.toPandas()

# Show data
display(unanalyzed_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing

# COMMAND ----------

def run_model_analysis(full_text: str) -> str:
    if not full_text:
        return 'Unrelated'
    
    # Ensure text is cut to size
    full_text = full_text[:512]

    # Run text through inference model
    result = analyzer(full_text)

    # Extract class label
    label = result[0]['label']
    return label

# COMMAND ----------

# Analyze all full text
unanalyzed_pd['label'] = unanalyzed_pd['full_text'].progress_apply(run_model_analysis)

# COMMAND ----------

# Filter to only tweets which are closure-related
filtered_pd = unanalyzed_pd[unanalyzed_pd['label']=='Closure-Related']
print(f"FILTERED DOWN TO {len(filtered_pd)} TWEETS")

# COMMAND ----------

filtered_pd.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export

# COMMAND ----------

# Convert to spark table
filtered = spark.createDataFrame(filtered_pd)

# Join with original df to get additional data
join_df = unanalyzed_df.select('id', 'attachments', 'attachments_text', 'attachment_img', 'attachment_targets') \
                      .withColumnRenamed('id', 'join_id')
filtered = filtered.join(join_df, filtered.id==join_df.join_id) \
                   .drop('join_id')

# COMMAND ----------

# DBTITLE 1,Format df for write
# Add columns as necessary
filtered = filtered.withColumn('filtered_at', f.current_timestamp()) \
                   .withColumn('analyzed_at', f.current_timestamp()) \
                   .withColumn('extracted_at', f.lit(None).cast('timestamp')) \
                   .withColumn('reported_at', f.lit(None).cast('timestamp')) \
                   .withColumn('closure_related', f.lit(True).cast('boolean')) \
                   .withColumn('closure_type', f.lit(None).cast('string')) \
                   .withColumn('closure_start_date', f.lit(None).cast('date')) \
                   .withColumn('closure_end_date', f.lit(None).cast('date')) \
                   .withColumn('closure_cause', f.lit(None).cast('string')) \
                   .withColumn('closure_entity_type', f.lit(None).cast('string')) \
                   .withColumn('closure_entity', f.lit(None).cast('string'))

# Select columns in order
filtered = filtered.select('id', 'created_time', 'filtered_at', 'analyzed_at', 'extracted_at', 'reported_at',
                           'facebook_account_id', 'permalink_url', 'message', 'full_text',
                           'attachments', 'attachments_text', 'attachment_img', 'attachment_targets',
                           'closure_related', 'closure_type', 'closure_start_date', 'closure_end_date',
                           'closure_cause', 'closure_entity_type', 'closure_entity'
                           )

filtered.display()

# COMMAND ----------

# Write to temp table for merge
filtered.createOrReplaceTempView('facebook_closure_temp')

# COMMAND ----------

# DBTITLE 1,Merge filtered posts to facebook_closure table
# MAGIC %sql
# MAGIC MERGE INTO schoolclosure_adl.facebook_closure dest
# MAGIC      USING (SELECT DISTINCT * FROM facebook_closure_temp) source
# MAGIC         ON (dest.id=source.id)
# MAGIC       WHEN NOT MATCHED
# MAGIC       THEN INSERT (id, created_time, filtered_at, analyzed_at, extracted_at, reported_at, facebook_account_id,
# MAGIC                    permalink_url, message, full_text, attachments, attachments_text, attachment_img, attachment_targets,
# MAGIC                    closure_related, closure_type, closure_start_date, closure_end_date, closure_cause, closure_entity_type, closure_entity)
# MAGIC            VALUES (id, created_time, filtered_at, analyzed_at, extracted_at, reported_at, facebook_account_id,
# MAGIC                    permalink_url, message, full_text, attachments, attachments_text, attachment_img, attachment_targets,
# MAGIC                    closure_related, closure_type, closure_start_date, closure_end_date, closure_cause, closure_entity_type, closure_entity);

# COMMAND ----------

# DBTITLE 1,Update facebook table with filtered_date
# MAGIC %sql
# MAGIC UPDATE schoolclosure_adl.facebook_filtered
# MAGIC     SET analyzed_at = CURRENT_TIMESTAMP()
# MAGIC   WHERE analyzed_at IS NULL;

# COMMAND ----------

# Exit notebook with success message
dbutils.notebook.exit('SUCCESS')

# COMMAND ----------


