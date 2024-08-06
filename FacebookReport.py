# Databricks notebook source
# MAGIC %md
# MAGIC # Facebook Report
# MAGIC
# MAGIC The purpose of this notebook is to read in the unreported closure-related posts from the database, match them with the best school/district from the database, and format a final report for the School Closure team.
# MAGIC
# MAGIC ## Inputs
# MAGIC * schoolclosure.facebook_closure TABLE
# MAGIC * schoolclosure.schools TABLE
# MAGIC * schoolclosure.districts TABLE
# MAGIC
# MAGIC ## Processes
# MAGIC
# MAGIC ## Outputs
# MAGIC * schoolclosure.facebook_report table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install fuzzywuzzy tqdm

# COMMAND ----------

# Standard Imports
import re
import os
import datetime as dt
import json
import pyspark.sql.functions as f
from datetime import datetime

# Third Party Imports
import pandas as pd
import requests
from tqdm import tqdm
from fuzzywuzzy import fuzz


# COMMAND ----------

# Setup tqdm
tqdm.pandas()

# COMMAND ----------

previous_days_to_scan = 100

today = dt.date.today() - dt.timedelta(previous_days_to_scan)
today = dt.datetime.strftime(today, '%Y-%m-%d')
print("DATE TO QUERY: ",  today)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prep Accounts For Join

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW school_info AS (
# MAGIC
# MAGIC WITH exploded_accounts AS (
# MAGIC   SELECT fa.id AS account_id,
# MAGIC          fa.fb_account_id AS fb_account_id,
# MAGIC          EXPLODE(fa.school_ids) AS school_id_exploded
# MAGIC     FROM schoolclosure_adl.facebook_accounts fa
# MAGIC    WHERE fa.school_ids IS NOT NULL
# MAGIC ),
# MAGIC   single_schools AS (
# MAGIC SELECT ea.account_id,
# MAGIC        ea.fb_account_id,
# MAGIC        sch.*
# MAGIC   FROM exploded_accounts ea
# MAGIC   JOIN schoolclosure_adl.schools_adj sch
# MAGIC     ON ea.school_id_exploded=sch.school_id
# MAGIC   )
# MAGIC   SELECT
# MAGIC     s.account_id AS account_id,
# MAGIC     s.fb_account_id AS fb_account_id,
# MAGIC     s.school_id AS school_id,
# MAGIC     NULL AS nces_id,
# MAGIC     'school' AS account_type,
# MAGIC     s.low_grade AS low_grade,
# MAGIC     s.high_grade AS high_grade,
# MAGIC     s.school_name AS school_name,
# MAGIC     s.district AS district_name,
# MAGIC     s.district_id AS district_id,
# MAGIC     s.county AS county,
# MAGIC     s.address AS address,
# MAGIC     s.city AS city,
# MAGIC     s.state AS state,
# MAGIC     s.zip AS zip,
# MAGIC     s.zip_4 AS zip_4,
# MAGIC     CAST(s.students AS INTEGER) AS num_students,
# MAGIC     s.website AS website 
# MAGIC   FROM single_schools s
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM school_info;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW district_info AS (
# MAGIC
# MAGIC WITH exploded_accounts AS (
# MAGIC   SELECT fa.id AS account_id,
# MAGIC          fa.fb_account_id AS fb_account_id,
# MAGIC          EXPLODE(fa.district_ids) AS district_id_exploded
# MAGIC     FROM schoolclosure_adl.facebook_accounts fa
# MAGIC    WHERE fa.district_ids IS NOT NULL
# MAGIC ),
# MAGIC   single_districts AS (
# MAGIC SELECT ea.account_id,
# MAGIC        ea.fb_account_id,
# MAGIC        dist.*
# MAGIC   FROM exploded_accounts ea
# MAGIC   JOIN schoolclosure_adl.districts dist
# MAGIC     ON ea.district_id_exploded=dist.nces_id
# MAGIC   )
# MAGIC   SELECT
# MAGIC     d.account_id AS account_id,
# MAGIC     d.fb_account_id AS fb_account_id,
# MAGIC     NULL AS school_id,
# MAGIC     d.nces_id AS nces_id,
# MAGIC     'district' AS account_type,
# MAGIC     NULL AS low_grade,
# MAGIC     NULL AS high_grade,
# MAGIC     NULL AS school_name,
# MAGIC     d.name AS district_name,
# MAGIC     d.nces_id AS district_id,
# MAGIC     NULL AS county,
# MAGIC     d.location_street1 AS address,
# MAGIC     d.location_city AS city,
# MAGIC     d.location_state AS state,
# MAGIC     d.location_zip AS zip,
# MAGIC     CAST(NULL AS FLOAT) AS zip_4,
# MAGIC     dc.num_students AS num_students,
# MAGIC     d.best_url AS website 
# MAGIC   FROM single_districts d
# MAGIC   JOIN (
# MAGIC                  SELECT schools_cast.district_id AS district_id,
# MAGIC                         SUM(schools_cast.num_students) AS num_students
# MAGIC                     FROM (
# MAGIC                         SELECT district_id,
# MAGIC                                 CAST(students AS INTEGER) AS num_students
# MAGIC                             FROM schoolclosure_adl.schools_adj
# MAGIC                     ) schools_cast
# MAGIC                 GROUP BY schools_cast.district_id
# MAGIC                 ) AS dc
# MAGIC     ON d.nces_id = dc.district_id
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM district_info LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW all_accounts AS (
# MAGIC   SELECT * FROM school_info
# MAGIC   UNION ALL
# MAGIC   SELECT * FROM district_info
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM all_accounts;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(DISTINCT account_id) FROM all_accounts;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM all_accounts;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM schoolclosure_adl.facebook_closure LIMIT 10;

# COMMAND ----------

# Ingest posts as dataFrame
select_statement = '''
         SELECT fc.id AS post_id,
                fc.created_time AS created_time,
                fc.facebook_account_id AS fb_account_id,
                aa.fb_account_id AS fb_account_name,
                fc.permalink_url AS permalink_url,
                fc.full_text AS full_text,
                fc.closure_type AS closure_type,
                fc.closure_start_date AS closure_start_date,
                fc.closure_end_date AS closure_end_date,
                fc.closure_cause AS closure_cause,
                LOWER(fc.closure_entity_type) AS closure_entity_type,
                fc.closure_entity AS closure_entity,
                aa.account_type AS account_type,
                aa.low_grade AS low_grade,
                aa.high_grade AS high_grade,
                aa.school_name AS school_name,
                aa.district_name AS district_name,
                aa.district_id AS district_id,
                aa.county AS county,
                aa.address AS address,
                aa.city AS city,
                aa.state AS state,
                aa.zip AS zip,
                aa.zip_4 AS zip_4,
                CAST(aa.num_students AS INTEGER) AS num_students,
                aa.website AS website
     FROM schoolclosure_adl.facebook_closure fc
     JOIN all_accounts aa
       ON fc.facebook_account_id=aa.account_id
    WHERE fc.reported_at IS NULL
'''
new_posts = spark.sql(select_statement)

# COMMAND ----------

new_posts.display()

# COMMAND ----------

# Count new posts
new_posts = new_posts.dropDuplicates(subset=['post_id'])
ingested_rows = new_posts.cache().count()

# If no records to process, exit
if ingested_rows==0:
    print("NO RECORDS TO PROCESS, EXITING")
    dbutils.notebook.exit("NO RECORDS TO PROCESS")
else:
    print(f"INGESTED {ingested_rows} NEW POSTS")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export

# COMMAND ----------

# Export to temp table
new_posts.createOrReplaceTempView('new_posts')

# COMMAND ----------

# DBTITLE 1,Merge Into Facebook_report
# MAGIC %sql
# MAGIC MERGE INTO schoolclosure_adl.facebook_report dest
# MAGIC  USING (SELECT * FROM new_posts) source
# MAGIC     ON (dest.post_id = source.post_id)
# MAGIC
# MAGIC  WHEN MATCHED THEN UPDATE
# MAGIC   SET dest.created_time = source.created_time,
# MAGIC            dest.fb_account_id = source.fb_account_id,
# MAGIC            dest.fb_account_name = source.fb_account_name,
# MAGIC            dest.permalink_url = source.permalink_url,
# MAGIC            dest.full_text = source.full_text,
# MAGIC            dest.closure_type = source.closure_type, 
# MAGIC            dest.closure_start_date = source.closure_start_date, 
# MAGIC            dest.closure_end_date = source.closure_end_date, 
# MAGIC            dest.closure_cause = source.closure_cause, 
# MAGIC            dest.closure_entity_type = source.closure_entity_type,
# MAGIC            dest.closure_entity = source.closure_entity,
# MAGIC            dest.account_type = source.account_type,
# MAGIC            dest.low_grade = source.low_grade,
# MAGIC            dest.high_grade = source.high_grade,
# MAGIC            dest.school_name = source.school_name,
# MAGIC            dest.district_name = source.district_name,
# MAGIC            dest.district_id = source.district_id,
# MAGIC            dest.county = source.county,
# MAGIC            dest.address = source.address,
# MAGIC            dest.city = source.city,
# MAGIC            dest.state = source.state,
# MAGIC            dest.zip = source.zip, 
# MAGIC            dest.zip_4 = source.zip_4,
# MAGIC            dest.num_students = source.num_students,
# MAGIC            dest.website = source.website
# MAGIC            
# MAGIC  WHEN NOT MATCHED 
# MAGIC  THEN INSERT (post_id, created_time, fb_account_id, fb_account_name, permalink_url, full_text,
# MAGIC                                 closure_type, closure_start_date, closure_end_date, closure_cause, closure_entity_type,
# MAGIC                                 closure_entity, account_type, low_grade, high_grade, school_name, district_name,
# MAGIC                                 district_id, county, address, city, state, zip, zip_4, num_students, website)
# MAGIC  VALUES (post_id, created_time, fb_account_id, fb_account_name, permalink_url, full_text,
# MAGIC                                 closure_type, closure_start_date, closure_end_date, closure_cause, closure_entity_type,
# MAGIC                                 closure_entity, account_type, low_grade, high_grade, school_name, district_name,
# MAGIC                                 district_id, county, address, city, state, zip, zip_4, num_students, website)
# MAGIC   ;

# COMMAND ----------

# DBTITLE 1,Merge Into Replica Table
# MAGIC %sql
# MAGIC MERGE INTO schoolclosure_adl.facebook_report_replica dest
# MAGIC  USING (SELECT * FROM new_posts) source
# MAGIC     ON (dest.post_id = source.post_id)
# MAGIC
# MAGIC  WHEN MATCHED THEN UPDATE
# MAGIC   SET dest.created_time = source.created_time,
# MAGIC            dest.fb_account_id = source.fb_account_id,
# MAGIC            dest.fb_account_name = source.fb_account_name,
# MAGIC            dest.permalink_url = source.permalink_url,
# MAGIC            dest.full_text = source.full_text,
# MAGIC            dest.closure_type = source.closure_type, 
# MAGIC            dest.closure_start_date = source.closure_start_date, 
# MAGIC            dest.closure_end_date = source.closure_end_date, 
# MAGIC            dest.closure_cause = source.closure_cause, 
# MAGIC            dest.closure_entity_type = source.closure_entity_type,
# MAGIC            dest.closure_entity = source.closure_entity,
# MAGIC            dest.account_type = source.account_type,
# MAGIC            dest.low_grade = source.low_grade,
# MAGIC            dest.high_grade = source.high_grade,
# MAGIC            dest.school_name = source.school_name,
# MAGIC            dest.district_name = source.district_name,
# MAGIC            dest.district_id = source.district_id,
# MAGIC            dest.county = source.county,
# MAGIC            dest.address = source.address,
# MAGIC            dest.city = source.city,
# MAGIC            dest.state = source.state,
# MAGIC            dest.zip = source.zip, 
# MAGIC            dest.zip_4 = source.zip_4,
# MAGIC            dest.num_students = source.num_students,
# MAGIC            dest.website = source.website
# MAGIC            
# MAGIC  WHEN NOT MATCHED 
# MAGIC  THEN INSERT (post_id, created_time, fb_account_id, fb_account_name, permalink_url, full_text,
# MAGIC                                 closure_type, closure_start_date, closure_end_date, closure_cause, closure_entity_type,
# MAGIC                                 closure_entity, account_type, low_grade, high_grade, school_name, district_name,
# MAGIC                                 district_id, county, address, city, state, zip, zip_4, num_students, website)
# MAGIC  VALUES (post_id, created_time, fb_account_id, fb_account_name, permalink_url, full_text,
# MAGIC                                 closure_type, closure_start_date, closure_end_date, closure_cause, closure_entity_type,
# MAGIC                                 closure_entity, account_type, low_grade, high_grade, school_name, district_name,
# MAGIC                                 district_id, county, address, city, state, zip, zip_4, num_students, website)
# MAGIC   ;

# COMMAND ----------

# DBTITLE 1,Update posts in facebook_closure
# MAGIC %sql
# MAGIC UPDATE schoolclosure_adl.facebook_closure
# MAGIC    SET reported_at = CURRENT_TIMESTAMP()
# MAGIC   WHERE reported_at IS NULL;

# COMMAND ----------

# Exit notebook with success message
dbutils.notebook.exit('SUCCESS')

# COMMAND ----------


