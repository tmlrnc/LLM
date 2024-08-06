# Databricks notebook source
# MAGIC %md
# MAGIC # Facebook Extract
# MAGIC
# MAGIC The purpose of this notebook is to read in the closure-related posts from the database, and extract standardized information using an LLM.
# MAGIC
# MAGIC ## Inputs
# MAGIC * schoolclosure.facebook_closure TABLE
# MAGIC
# MAGIC ## Processes
# MAGIC
# MAGIC ## Outputs
# MAGIC * schoolclosure.facebook_closure fields:
# MAGIC   * `extracted_at`
# MAGIC   * `closure_type`
# MAGIC   * `closure_start_date`
# MAGIC   * `closure_end_date`
# MAGIC   * `closure_cause`
# MAGIC   * `closure_entity_type`
# MAGIC   * `closure_entity`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install typing-inspect==0.8.0 pydantic==1.10.11 openai==1.6.1 langchain==0.0.353 typing_extensions typing_extensions azure-identity

# COMMAND ----------

# Standard Imports
import re
import os
import datetime as dt
import json
import time
import pyspark.sql.functions as f
from pyspark.sql.types import *
from datetime import datetime

# Third Party Imports
import pandas as pd
import requests
from tqdm import tqdm
import openai
from openai import RateLimitError, BadRequestError, AuthenticationError
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from azure.identity import ClientSecretCredential

# COMMAND ----------

# Setup tqdm
tqdm.pandas()

# COMMAND ----------

# The values below can be found in the Overview page of your Azure Service Principal (the client secret is not fully visible and is usually stored in an Azure key vault)
credential = ClientSecretCredential(tenant_id="9ce70869-60db-44fd-abe8-d2767077fc8f", client_id="b16c8490-09fc-45dd-8fdf-09eba1923eab", client_secret=dbutils.secrets.get(scope="dbs-scope-EDAV-DEV-VAULT", key="CDC-SchoolClosure"))

# Set the AZURE_OPENAI_API_KEY to the token from the Azure credential
os.environ["AZURE_OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token

# Set endpoint
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://edav-openai-east1-2.openai.azure.com/"

# COMMAND ----------

# Instantiate llm via langchain
llm = AzureChatOpenAI(
    openai_api_version="2024-02-15-preview",
    azure_deployment="GPT35-Turbo-0613",
)

# COMMAND ----------

# Test llm functionality
llm([HumanMessage(content='Tell me a joke.')])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest

# COMMAND ----------

# DBTITLE 1,Ingest Unprocessed Posts
# Ingest from hive metastore
unprocessed_df = spark.sql('SELECT * FROM schoolclosure_adl.facebook_closure WHERE extracted_at IS NULL;')
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

# MAGIC %md
# MAGIC ### closure_type

# COMMAND ----------

def check_post_type(llm, post_text):
    template = f"""You are analyzing tweets from a school system. Given the following tweet, what category of school closure is being discussed? Use the examples below as a guide:

        Question: "Due to the winter weather, Kelly Mill elementary will be closed on Tuesday, April 3rd." Is the answer: A. School is completely closed, B. Late start or early dismissal, C. Virtual/remote/asynchronous learning, D. Hybrid/blended learning, or E. Cannot tell?
        Answer: A. School is completely closed

        Question: "Due to weather conditions, TODAY will be a Flexible Instructional Day (FID). School will be virtual for all students. See the school website for more info." Is the answer: A. School is completely closed, B. Late start or early dismissal, C. Virtual/remote/asynchronous learning, D. Hybrid/blended learning, or E. Cannot tell?
        Answer: C. Virtual/remote/asynchronous learning

        Question: "Due to the current weather conditions, ASBVI will be dismissed at 1:00 p.m. today." Is the answer: A. School is completely closed, B. Late start or early dismissal, C. Virtual/remote/asynchronous learning, D. Hybrid/blended learning, or E. Cannot tell?
        Answer: B. Late start or early dismissal

        Question: "{post_text}" Is the answer: A. School is completely closed, B. Late start or early dismissal, C. Virtual/remote/asynchronous learning, D. Hybrid/blended learning, or E. Cannot tell?
        Answer: 
        """
    try:
        response = llm([HumanMessage(content=template)])

        return response.content
    
    except (RateLimitError, AuthenticationError) as e:
        # Sleep then try again
        print("ERROR - SLEEPING FOR 1 MINUTE BEFORE RETRY")
        time.sleep(61)
        response = llm([HumanMessage(content=template)])

        return response.content   
    
    # Handling common exceptions gracefully
    # ValueError occurs when no answer is returned
    # BadRequestError when content filtering is triggered by a prompt
    except (ValueError, BadRequestError) as e:

        return None

def interpreter_closure_type(response: str) -> str:
    """
    Reads LLM response, parses a standardized categorical answer for closure type.
    
    Parameters
    ----------
    response
        str - LLM response to prompt
        
    Returns
    -------
    str
        standardized category based on logic
    """
    if not isinstance(response, str):
        return None

    if ("A. " in response) or ("completely" in response):
        return "closure"
    elif ("B. " in response) or ("early" in response):
        return "partial closure"
    elif ("C. " in response) or ("online" in response):
        return "virtual"
    elif ("D. " in response) or ("Hybrid" in response):
        return "hybrid"
    else:
        return "closure"

# %% ../nbs/07_twitter_extract.ipynb 15
def apply_check_post_type(df: pd.DataFrame, llm, testing: bool = False) -> pd.DataFrame:
    """
    Iterate through dataframe, taking each tweet, formatting into a prompt, querying LLM and standardizing response.
    Categoriezes closure type of given tweets.
    
    Parameters
    ----------
    df
        pd.DataFrame - dataframe of tweets stored in 'full_text' column
    llm
        langchain model to query
    testing
        boolean - whether to print prompt/response, default is False

        
    Returns
    -------
    df
        pd.DataFrame - containg all previous data as well as 'closure_type' category
    """

    # Check tweet type for all values
    df['response'] = df.progress_apply(lambda x: check_post_type(
                                        llm,
                                        x['full_text']
                                        ), axis=1)

    # Interpret response
    df['closure_type'] = df['response'].apply(interpreter_closure_type)

    # Drop response
    df = df.drop(columns=['response'])

    return df


# COMMAND ----------

processed_pd = apply_check_post_type(unprocessed_pd, llm, testing=True)

# COMMAND ----------

processed_pd.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### dates

# COMMAND ----------

def check_post_dates(llm, post_text, post_date, testing = False):
    """
    Extract start_date and end_date from a given post using langchain model.
    If date cannot be determined, defaults to current date.
    
    Parameters
    ----------
    llm
        langchain LLM wrapper
    post
        str - text of post
    post_date
        datetime - datetime.date of created_time for post
    testing
        bool - whether or not to print details during execution. Default = False
        
    Returns
    -------
    start_date
        string of closure start date
    end_date
        string of closure end date
    """

    # Parse date strings
    # today = dt.date.today()
    cur_day = post_date.strftime('%A')
    # cur_date = today - dt.timedelta(days=1)
    cur_date = post_date.strftime('%m/%d/%Y')
    tom_date = post_date + dt.timedelta(days=1)
    tom_date = tom_date.strftime('%m/%d/%Y')
    cur_year = cur_date[-4:]

    cur_year = cur_date[-4:]

    post_prompt = f"""Today's date is {cur_day}, {cur_date}. You are analyzing facebook posts from a school system. Given a specific post, what dates are affected by the closure? Use the examples below as a guide:

    Input: In preparation of #Eta, due to the dramatic shift in the path of Tropical Storm Eta, school is canceled tomorrow as are all after school activities. https://t.co/FUPLv0BxEa.
    Output: {tom_date}-{tom_date}

    Input: Due to unforeseen circumstances, school is canceled today, as are all after school activities.
    Output: {cur_date}-{cur_date}

    Input: NIblack Elementary School will be closed March 23-27. The food program pickup point will now be held at Richardson Community Center from 10am-12pm.
    Output: 03/23/{cur_year}-03/27/{cur_year}

    Input: {post_text}
    Output: 
    """
    try:
        if testing:
            print("POST: ", post_text)
            post_response = llm([HumanMessage(content=post_prompt)])
            print("RESPONSE: ", post_response)
            print("----------------------------------------------")
            return post_response.content
        else:
            post_response = llm([HumanMessage(content=post_prompt)])
            return post_response.content
        
    except (RateLimitError, AuthenticationError) as e:
        # Sleep then try again
        print("ERROR - SLEEPING FOR 1 MINUTE BEFORE RETRY")
        time.sleep(61)
        post_response = llm([HumanMessage(content=post_prompt)])

        return post_response.content 
        
    # Handling common exceptions gracefully
    # ValueError occurs when no answer is returned
    # BadRequestError when content filtering is triggered by a prompt
    except (ValueError, BadRequestError) as e:

        return None

def parse_dates(date_response: str, cur_date: str):
    """
    Parse LLM response to extract start and end date in standardized format
    
    Parameters
    ----------
    date_response
        str of LLM response to date query
    cur_date
        string of current date, in 'YYYY-MM-DD' format
        
    Returns
    -------
    start_date
        string of closure start date
    end_date
        string of closure end date
        
    """
    if not isinstance(date_response, str):
        return None, None

    start_date = cur_date
    end_date = cur_date
    if ("/" not in date_response) or (date_response=="N/A"):
        return start_date, end_date
    elif "-" not in date_response:
        return start_date, end_date
    else:
        dates = date_response.split("-")
        if "/" in dates[0]:
            start_date = dates[0].split()[0]
        if "/" in dates[1]:
            end_date = dates[1].split()[0]
        return start_date, end_date

def apply_check_post_dates(df: pd.DataFrame, llm, testing: bool = False) -> pd.DataFrame:
    """
    Iterate through dataframe, taking each post, extracting closure start and end date.
    Categoriezes closure date of given posts.
    
    Parameters
    ----------
    df
        pd.DataFrame - dataframe of tweets stored in 'text' column
    llm
        langchain llm to query
    testing
        boolean - whether to print prompt/response, default is False

        
    Returns
    -------
    df
        pd.DataFrame - containg all previous data as well as 'closure_start_date' and 'closure_end_date' columns
    """

    # Check tweet type for all values
    df['date_response'] = df.progress_apply(lambda x: check_post_dates(
                                        llm,
                                        x['full_text'],
                                        x['created_date'],
                                        testing=testing
                                        ), axis=1)
    
    # Interpret response
    cur_date = dt.date.today()
    cur_date = cur_date.strftime('%m/%d/%Y')
    df['closure_start_date'], df['closure_end_date'] = zip(*df['date_response'].apply(parse_dates, cur_date=cur_date))

    # Drop date response
    df = df.drop(columns=['date_response'])

    return df


# COMMAND ----------

# Generate created_date info
processed_pd['created_date'] = processed_pd['created_time'].dt.date

# COMMAND ----------

processed_pd = apply_check_post_dates(processed_pd, llm, testing=True)

# COMMAND ----------

processed_pd.head()

# COMMAND ----------

processed_pd.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### closure_type

# COMMAND ----------

def interpreter_closure_type(response: str) -> str:
    """
    Reads LLM response, parses a standardized categorical answer for closure type.
    
    Parameters
    ----------
    response
        str - LLM response to prompt
        
    Returns
    -------
    str
        standardized category based on logic
    """
    if not isinstance(response, str):
        return None

    if ("A. " in response) or ("completely" in response):
        return "closure"
    elif ("B. " in response) or ("early" in response):
        return "partial closure"
    elif ("C. " in response) or ("online" in response):
        return "virtual"
    elif ("D. " in response) or ("Hybrid" in response):
        return "hybrid"
    else:
        return "closure"

# %% ../nbs/07_twitter_extract.ipynb 14
def check_post_type(post: str, llm, testing: bool = False) -> str:
    """
    Take a given post, generate a prompt, query the LLM.
    
    Parameters
    ----------
    post
        str - post to analyze
    llm
        langchain llm to be queried
    testing
        boolean - whether to print prompt/response, default is False
        
    Returns
    -------
    post_response
        str - raw text response from LLM
    """

    # Build template for system prompt
    post_prompt = f"""You are analyzing facebook posts from a school system. Given the following post, what category of school closure is being discussed? Use the examples below as a guide:

    Question: "Due to the winter weather, Kelly Mill elementary will be closed on Tuesday, April 3rd." Is the answer: A. School is completely closed, B. Late start or early dismissal, C. Virtual/remote/asynchronous learning, D. Hybrid/blended learning, or E. Cannot tell?
    Answer: A. School is completely closed

    Question: "Due to weather conditions, TODAY will be a Flexible Instructional Day (FID). School will be virtual for all students. See the school website for more info." Is the answer: A. School is completely closed, B. Late start or early dismissal, C. Virtual/remote/asynchronous learning, D. Hybrid/blended learning, or E. Cannot tell?
    Answer: C. Virtual/remote/asynchronous learning

    Question: "Due to the current weather conditions, ASBVI will be dismissed at 1:00 p.m. today." Is the answer: A. School is completely closed, B. Late start or early dismissal, C. Virtual/remote/asynchronous learning, D. Hybrid/blended learning, or E. Cannot tell?
    Answer: B. Late start or early dismissal

    Question: "{post}" Is the answer: A. School is completely closed, B. Late start or early dismissal, C. Virtual/remote/asynchronous learning, D. Hybrid/blended learning, or E. Cannot tell?
    Answer: 
    """
    try:
        if testing:
            print("POST: ", post)
            post_response = llm([HumanMessage(content=post_prompt)])
            print("RESPONSE: ", post_response)
            print("----------------------------------------------")
            return post_response.content
        else:
            post_response = llm([HumanMessage(content=post_prompt)])
            return post_response.content
        
    except (RateLimitError, AuthenticationError) as e:
        # Sleep then try again
        print("ERROR - SLEEPING FOR 1 MINUTE BEFORE RETRY")
        time.sleep(61)
        response = llm([HumanMessage(content=post_prompt)])

        return response.content 
    
    # Handling common exceptions gracefully
    # ValueError occurs when no answer is returned
    # BadRequestError when content filtering is triggered by a prompt
    except (ValueError, BadRequestError) as e:

        return None

def apply_check_post_type(df: pd.DataFrame, llm, testing: bool = False) -> pd.DataFrame:
    """
    Iterate through dataframe, taking each tweet, formatting into a prompt, querying LLM and standardizing response.
    Categoriezes closure type of given tweets.
    
    Parameters
    ----------
    df
        pd.DataFrame - dataframe of tweets stored in 'text' column
    llm
        langchain llm to be queried
    testing
        boolean - whether to print prompt/response, default is False

        
    Returns
    -------
    df
        pd.DataFrame - containg all previous data as well as 'closure_type' category
    """

    # Check tweet type for all values
    df['response'] = df.progress_apply(lambda x: check_post_type(
                                        x['full_text'],
                                        llm,
                                        testing=testing
                                        ), axis=1)

    # Interpret response
    df['closure_type'] = df['response'].apply(interpreter_closure_type)

    # Drop date response
    df = df.drop(columns=['response'])

    return df


# COMMAND ----------

processed_pd = apply_check_post_type(processed_pd, llm, testing=True)

# COMMAND ----------

processed_pd.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### closure_cause

# COMMAND ----------

def interpreter_closure_reason(response: str) -> str:
    """
    Reads LLM response, parses a standardized categorical answer for closure reason.
    
    Parameters
    ----------
    response
        str - LLM response to prompt
        
    Returns
    -------
    str
        standardized category based on logic
    """
    if not isinstance(response, str):
        return None

    # Check for marker statement
    if 'The reason given is' in response:
        split_response = response.split('given is ')
        # Get text immediately following this phrase
        reason_text = split_response[1]
        # Split to get first word of response
        reason_text_split = reason_text.split()
        reason_given = reason_text_split[0]
    else:
        return None

    if 'health-related' in reason_given.lower():
        return 'health'
    elif 'weather-related' in reason_given.lower():
        return 'weather'
    elif 'facilities-related' in reason_given.lower():
        return 'facilities'
    elif 'security-related' in reason_given.lower():
        return 'security'
    else:
        return None

def check_post_reason(post: str, llm, testing: bool = False) -> str:
    """
    Submit a post to given llm, return response.
    
    Parameters
    ----------
    tweet
        str - text of given tweet
    llm
        lanchain llm object to be queried
    testing
        bool - verbose prints during execution, default is False
        
    Returns
    -------
    str
        full response from llm
    """

    # Build template for system prompt
    post_prompt = f"""You are analyzing facebook posts from a school system. Given the following post, would you classify the reason given for the change as health-related, weather-related, facilities-related, security-related or other? If you can't tell, just say so, don't make up an answer. Use the examples below as a guide:

    Tweet: "Due to the winter weather, Kelly Mill elementary will be closed on Tuesday, April 3rd."
    Answer: The reason given is weather-related.

    Tweet: "Commissioner Penny Schwinn released this statement regarding the Governors recommendation to extend school closures through the end of the school year and the creation of the of the COVID-19 Child Wellbeing Task Force. https://t.co/7xaxv30a8d"
    Answer: The reason given is health-related.

    Tweet: "We have too many teachers and bus drivers out with the flu today, so ASBVI will be moving to virtual instruction today."
    Answer: The reason given is health-related.

    Tweet: "All district schools are closed today, please stay tuned for more information."
    Answer: I cannot tell.

    Tweet: "The forecast is calling for icy, dangerous conditions in the morning. Because of this, we will have an asynchronous learning day tomorrow. Stay safe!"
    Answer: The reason given is weather-related.

    Tweet: "{post}"
    Answer: 
    """
    try:
        if testing:
            print("POST: ", post)
            post_response = llm([HumanMessage(content=post_prompt)])

            print("RESPONSE: ", post_response)
            print("----------------------------------------------")
            return post_response.content
        else:
            post_response = llm([HumanMessage(content=post_prompt)])
            return post_response.content
        
    except (RateLimitError, AuthenticationError) as e:
        # Sleep then try again
        print("ERROR - SLEEPING FOR 1 MINUTE BEFORE RETRY")
        time.sleep(61)
        post_response = llm([HumanMessage(content=post_prompt)])

        return post_response.content 
    
    # Handling common exceptions gracefully
    # ValueError occurs when no answer is returned
    # BadRequestError when content filtering is triggered by a prompt
    except (ValueError, BadRequestError) as e:

        return None

def apply_check_post_reason(df: pd.DataFrame, llm, testing: bool = False) -> pd.DataFrame:
    """
    Iterate through dataframe, taking each post, formatting into a prompt, querying LLM and standardizing response.
    Categoriezes closure reason of given posts.
    
    Parameters
    ----------
    df
        pd.DataFrame - dataframe of posts stored in 'full_text' column
    llm
        lanchain llm object to be queried
    testing
        boolean - whether to print prompt/response, default is False

        
    Returns
    -------
    df
        pd.DataFrame - containg all previous data as well as 'closure_cause' category
    """

    # Check tweet type for all values
    df['reason_response'] = df.progress_apply(lambda x: check_post_reason(
                                        x['full_text'],
                                        llm,
                                        testing=testing
                                        ), axis=1)
    
    # Interpret response
    df['closure_cause'] = df['reason_response'].apply(interpreter_closure_reason)

    # Drop reason response
    df = df.drop(columns=['reason_response'])

    return df

# COMMAND ----------

processed_pd = apply_check_post_reason(processed_pd, llm, testing=True)

# COMMAND ----------

processed_pd.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### closure_entity

# COMMAND ----------

def check_post_reference(post: str, llm, testing: bool = False) -> str:
    """
    Submit a post to given llm, return response.
    
    Parameters
    ----------
    post
        str - text of given post
    llm
        lanchain llm object to be queried
    testing
        bool - verbose prints during execution, default is False
        
    Returns
    -------
    str
        full response from llm
    """

    # Build template for system prompt
    post_prompt = f"""You are analyzing facebook posts from a school system. Given a specific post, is the entire district closed, or just a specific school? Please name the district or school if possible. Use the example below as a guide:

    Question: "Wyoming Public Schools will be closed Friday, November 18th. Stay home safe and warm! #BetterTogetherWPS #WyomingWolves https://t.co/fADGxybEo0"
    Answer: District. Wyoming Public Schools.

    Question: South Elementary dismissing at 2pm due to a power outage.
    Answer: School. South Elementary.

    Question: The district is CLOSED today (Monday, Sept. 5). https://t.co/ApMFoUU8Ak
    Answer: District.

    Question: Update: All CCSD schools are CLOSED tomorrow, Feb. 22, 2019. https://t.co/E8nAZLqFue
    Answer: District. CCSD.

    Question: Northern Local will be closed tomorrow, Jan. 19th.   Please be safe when traveling on icy road conditions.
    Answer: School. Northern Local.

    Question: Abby Kelley Foster will BE CLOSED today, Thursday, February 6, 2020 due to inclement weather. Stay safe!
    Answer: School. Abby Kelley Foster.

    Question: Happy snow day! Stay safe &amp; warm https://t.co/U8p24cq1X3
    Answer: Unknown.

    Question: {post}
    Answer: 
    """
    try:
        if testing:
            print("POST: ", post)
            post_response = llm([HumanMessage(content=post_prompt)])
            print("REPSONSE: ", post_response)
        
        else:
            post_response = llm([HumanMessage(content=post_prompt)])
        return post_response.content
    
    except (RateLimitError, AuthenticationError) as e:
        # Sleep then try again
        print("ERROR - SLEEPING FOR 1 MINUTE BEFORE RETRY")
        time.sleep(61)
        post_response = llm([HumanMessage(content=post_prompt)])

        return post_response.content 
    
    # Handling common exceptions gracefully
    # ValueError occurs when no answer is returned
    # BadRequestError when content filtering is triggered by a prompt
    except (ValueError, BadRequestError) as e:

        return None

def split_reference_response(reference: str):
    """
    Reads LLM response, parses the closure entity and entity type.
    
    Parameters
    ----------
    response
        str - LLM response to prompt
        
    Returns
    -------
    closure_entity_type
        str - either 'district' or 'school'
    closure_entity
        str - name of district or school
    """

    if not isinstance(reference, str):
        return None, None

    # Select response after first assistant prompt
    try:
        reference = reference.split('### Assistant: ')[1]
    except IndexError:
        reference = reference

    if "Unknown" in reference:
        return None, None
    
    if '.' in reference:
        split_reference = [x.strip().replace(".", "") for x in reference.split(".")]
        # If two references are given
        if len(split_reference) > 1:
            if "No information" in split_reference[1]:
                return split_reference[0], None
            else:
                return split_reference[0], split_reference[1]
        else:
            # Handle no information response
            if "No information" in split_reference[0]:
                return None, None
            # Return only district/school identifier
            else:
                return split_reference[0], None
    else:
        return None, None

# %% ../nbs/07_twitter_extract.ipynb 47
def apply_check_post_reference(df: pd.DataFrame, llm, testing: bool = False) -> pd.DataFrame:
    """
    Iterate through dataframe, taking each post, formatting into a prompt, querying LLM and standardizing response.
    Categorizes closure entity as 'school' or 'district', and determines the entity if named.
    
    Parameters
    ----------
    df
        pd.DataFrame - dataframe of tweets stored in 'text' column
    llm
        langchain llm object to be queried
    testing
        boolean - whether to print prompt/response, default is False

        
    Returns
    -------
    df
        pd.DataFrame - containg all previous data as well as 'closure_entity_type' and 'closure_entity' categories
    """

    # Check post type for all values
    df['entity_response'] = df.progress_apply(lambda x: check_post_reference(
                                        x['full_text'],
                                        llm,
                                        testing=testing
                                        ), axis=1)
    
    # Interpret response
    df['closure_entity_type'], df['closure_entity'] = zip(*df['entity_response'].apply(split_reference_response))

    # Drop response
    df = df.drop(columns=['entity_response'])

    return df

# COMMAND ----------

processed_pd = apply_check_post_reference(processed_pd, llm, testing=True)

# COMMAND ----------

processed_pd.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export

# COMMAND ----------

# Convert to datetime
processed_pd['closure_start_date'] = pd.to_datetime(processed_pd['closure_start_date'], errors='coerce')
processed_pd['closure_end_date'] = pd.to_datetime(processed_pd['closure_end_date'], errors='coerce')

# COMMAND ----------

processed_pd.head()

# COMMAND ----------

# DBTITLE 1,Export dataframe to temp table
processed_df_schema = StructType([
        StructField('created_time', TimestampType(), False),
        StructField('id', StringType(), False),
        StructField('message', StringType(), True),
        StructField('permalink_url', StringType(), True),
        StructField('full_text', StringType(), True),
        StructField('facebook_account_id', IntegerType(), False),
        StructField('closure_type', StringType(), True),
        StructField('created_date', DateType(), True),
        StructField('closure_start_date', DateType(), True),
        StructField('closure_end_date', DateType(), True),
        StructField('closure_cause', StringType(), True),
        StructField('closure_entity_type', StringType(), True),
        StructField('closure_entity', StringType(), True)
        ]
    )

# Convert to spark table
processed_df = spark.createDataFrame(processed_pd, schema=processed_df_schema)

# Format date columns
processed_df = processed_df.withColumn('closure_start_date', f.col('closure_start_date').cast('date')) \
                           .withColumn('closure_end_date', f.col('closure_end_date').cast('date'))

# Add columns as necessary
processed_df = processed_df.withColumn('extracted_at', f.current_timestamp())

# Deduplicate
processed_df = processed_df.dropDuplicates(subset=['id'])

# Save as temp table
processed_df.createOrReplaceTempView('processed_merge')

# COMMAND ----------

processed_df.display()

# COMMAND ----------

# DBTITLE 1,Merge records into facebook_closure
# MAGIC %sql
# MAGIC   MERGE INTO schoolclosure_adl.facebook_closure target
# MAGIC                     USING processed_merge source
# MAGIC                         ON source.id = target.id
# MAGIC                 WHEN MATCHED THEN UPDATE
# MAGIC                     SET target.extracted_at = source.extracted_at,
# MAGIC                         target.closure_type = source.closure_type,
# MAGIC                         target.closure_start_date = source.closure_start_date,
# MAGIC                         target.closure_end_date = source.closure_end_date,
# MAGIC                         target.closure_cause = source.closure_cause,
# MAGIC                         target.closure_entity_type = source.closure_entity_type,
# MAGIC                         target.closure_entity = source.closure_entity;

# COMMAND ----------

# Exit notebook with success message
dbutils.notebook.exit('SUCCESS')

# COMMAND ----------


