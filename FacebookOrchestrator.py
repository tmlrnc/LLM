# Databricks notebook source
# MAGIC %md
# MAGIC # Facebook Orchestrator
# MAGIC
# MAGIC This notebook runs the entirety of the facebook analysis pipeline for the schoolclosure reporting system.
# MAGIC
# MAGIC ## Notebooks Run
# MAGIC
# MAGIC * 'FacebookFilter'
# MAGIC * 'FacebookAnalyze'

# COMMAND ----------

import time
def run_with_retry(notebook, timeout, args = {}, max_retries = 3):
  num_retries = 0
  while True:
    try:
      return dbutils.notebook.run(notebook, timeout, args)
    except Exception as e:
      if num_retries > max_retries:
        dbutils.notebook.exit(f'RUN FAILED AT {notebook}')
      else:
        print("Retrying error", e)
        time.sleep(60)
        num_retries += 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Facebook Ingest
# MAGIC
# MAGIC Ingests new facebook posts from all accounts.

# COMMAND ----------

# Run facebook ingest
run_with_retry("./FacebookIngest", timeout=10000, args = {}, max_retries = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Facebook Filter
# MAGIC
# MAGIC Ingests and filters all new facebook posts.

# COMMAND ----------

# Run facebook filtering
run_with_retry("./FacebookFilter", timeout=10000, args = {}, max_retries = 3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Facebook Analyze
# MAGIC
# MAGIC Ingests all newly filtered posts, analyzes them using pretrained roBERTa model.

# COMMAND ----------

# Run facebook analysis
run_with_retry("./FacebookAnalyze", timeout=10000, args = {}, max_retries = 3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Facebook Extract
# MAGIC
# MAGIC Ingests only closure-related posts, extracts information using Large Language Model.

# COMMAND ----------

# Run facebook extraction
run_with_retry("./FacebookExtract", timeout=20000, args = {}, max_retries = 3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Facebook Report
# MAGIC
# MAGIC Compile final report table for export/analysis.

# COMMAND ----------

# Run facebook report
run_with_retry("./FacebookReport", timeout=10000, args = {}, max_retries = 3)

# COMMAND ----------

dbutils.notebook.exit('SUCCESS')

# COMMAND ----------


