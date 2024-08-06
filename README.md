# Schoolclosure Databricks Pipeline

## Overview

The Schoolclosure Databricks Pipeline is designed to automate the process of tracking unplanned school closures in the United States. It leverages the power of Databricks, Machine Learning, Generative AI, and the Facebook Graph API to process and analyze large volumes of data related to school closures.

## Pipeline

The automated pipeline is initiated via a daily Databricks Workflow which triggers the FacebookOrchestrator notebook. This notebook calls other downstream notebooks in succession to execute the full pipeline. Those notebooks include:

- FacebookIngest: Retreiving all new posts from schools whose Facebook account is stored in the database. This notebook leverages the Facebook Graph API via a REST API call.
- FacebookFilter: Filtering all newly retrieved posts via a regex filter, quickly eliminating completely irrelevant posts.
- FacebookAnalyze: Uses a fine-tuned BERT-based model in pytorch to classify each remaining post as related or unrelated to unplanned school closure.
    - see model_card.md for more information about the model.
- FacebookExtract: For posts tagged as related to unplanned school closure, passes each post to a LLM via the Azure OpenAI API (gpt-3.5-turbo), requesting standardized information via crafted prompts.
- FacebookReports: Final tagged posts are combined with metadata about the school/district from their posting account, then all relevant data is posted to the final table for daily report.

### Pipeline Flow

![Pipeline Flow Diagram](./img/architecture_flow.png "Pipeline Overview")

### Architecture Diagram

![Architecture Diagram](./img/architecture.png "Architecture Diagram")

## Dashboard

The SchoolClosure dashboard is generated in Databricks using the SQL queries found in the `./sql/` directory. This dashboard runs every morning, following the pipeline run, and sends an automated email to subscribers.

## Data Storage

All data is stored in Databricks in the `schoolclosure_adl` schema. These tables can be recreated using the queries found in the `./sql/create_tables.sql` file.

The data found in the `schools` and `districts` tables was pulled from the NCED school finder website. Facebook accounts were detected in an automated manner using the `./utilities/SchoolClosure_FacebookAccountFinder.py` notebook.

## Necessary keys

In order to operate this pipeline, you need to have several authorization keys. Those are as follows:

* Facebook Graph API: To retrieve posts from public facebook pages, you need to have an API key with the "Page Public Content Access" permission enabled. This requires permission from Meta via Meta's Developer portal.
* Azure OpenAI: The pipeline is designed to interact with Azure OpenAI via a specific deployment, with keys stored as Databricks secrets. You can easily convert to a different LLM using the various connectors found in the [langchain python library](https://www.langchain.com/). 

## Getting Started

To get started with the Schoolclosure Databricks Pipeline, follow these steps:

1. Clone the repository in your databricks workspace.
2. Obtain necessary keys and populate in databricks secrets.
3. Create all tables in Databricks database.
4. Populate `schools` and `districts` tables using data from NCES.

## Contact Us

If you have any questions, contact the Tech R&D Lab at [techlab@cdc.gov](mailto:techlab@cdc.gov)