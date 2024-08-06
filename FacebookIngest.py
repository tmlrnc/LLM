# Databricks notebook source
# MAGIC %md
# MAGIC # Facebook Ingest
# MAGIC
# MAGIC The purpose of this notebook is to read in the latest facebook posts from the facebook Graph API.
# MAGIC
# MAGIC ## Inputs
# MAGIC * schoolclosure.facebook_accounts TABLE
# MAGIC
# MAGIC ## Processes
# MAGIC * Obtain current facebook graph api key
# MAGIC * Find most current post date for each account in table
# MAGIC * Search through each account, if new posts exists, extract and download to datalake
# MAGIC
# MAGIC ## Outputs
# MAGIC * schoolclosure.facebook_posts
# MAGIC * schoolclosure.facebook_accounts (last post date, id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %pip install tqdm

# COMMAND ----------

# Standard Imports
import re
import os
import datetime as dt
import json
import time
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.sql.types import *
from multiprocessing.pool import ThreadPool
import datetime
from urllib.request import urlopen

# Third Party Imports
import pandas as pd
import requests
from tqdm import tqdm

# COMMAND ----------

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# COMMAND ----------

# Setup tqdm
tqdm.pandas()

# COMMAND ----------

# Ingest facebook token
facebook_token = dbutils.secrets.get(scope = "schoolclosure", key = "facebooktoken")

# COMMAND ----------

# DBTITLE 1,Get current datetime timestamp for traceability
current_ts = dt.datetime.now()
current_dt = current_ts.isoformat(timespec='milliseconds')
print("CURRENT TIMESTAMP: ", current_dt)

# COMMAND ----------

# Set environment flag to 'dev' for print statements
# Conserves job cluster memory, prevents crashes
env = 'prod'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest

# COMMAND ----------

# Test ingest
account_id = 'kellymilles'
request_fields = 'created_time,id,message,permalink_url,attachments{description,media,target,title,url}'
since_timestamp_dt = datetime.datetime.strptime('2024-02-19T13:40:06+0000', "%Y-%m-%dT%H:%M:%S+0000")
since_timestamp = since_timestamp_dt.timestamp()
token = facebook_token
request_url = f'https://graph.facebook.com/v18.0/{account_id}/feed?fields={request_fields}&since={since_timestamp}&access_token={token}'
print("URL REQUESTED: ", request_url)

# with urlopen(request_url) as response:
# 	content = response.read()

# response = content.decode("utf-8", "ignore")

response = requests.get(request_url, verify=False)
response = response.text

print("Response:", response[0:100], '...')

# COMMAND ----------

# DBTITLE 1,Ingest Account Data
# Ingest from hive metastore
accounts_df = spark.sql('SELECT * FROM schoolclosure_adl.facebook_accounts WHERE fb_account_id IS NOT NULL AND LENGTH(fb_account_id) > 0 AND account_error IS FALSE;')
accounts_rows = accounts_df.cache().count()
print(f"INGESTED {accounts_rows} FACEBOOK ACCOUNTS")

# If no records to process, exit
if accounts_rows==0:
    print("NO RECORDS TO PROCESS, EXITING")
    dbutils.notebook.exit("ERROR NO FACEBOOK ACCOUNTS TO PROCESS")

# COMMAND ----------

# Convert to pandas dataframe
accounts_pd = accounts_df.toPandas()

# Show data
if env=='dev':
    accounts_pd.head(100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing

# COMMAND ----------

def request_with_retry(request_url: str, max_retries: int = 5) -> dict:
    """Retry using requests module with automatic retry if exception is thrown.

    Args:
        request_url (string): string of url to hit
        max_retries (int) : number of retries before exception thrown, default is 5

    Returns:
        dict of response from requests
    """
    # Make number of requests required
    for n in range(max_retries):
        try:
            response = requests.get(request_url, verify=False)
            # with urlopen(request_url) as response:
            #     content = response.read()
            #     text = content.decode("utf-8", "ignore")
            # return text
            return response
        
        except Exception as e:
            print(f"EXCEPTION on retry {n+1}: ", str(e))
            # Sleep for increasing backoff
            delay = 2 * (2 ** attempts)
            time.sleep(delay)
            # Continue to next retry
            pass

    raise Exception('Max Retries Exceeded')

def graph_api_get(request_url: str, account_name: str, account_id: int) -> dict:
    """Get results from Facebook Graph API

    Args:
        request_url (string) : string of url to hit
        account_name (string) : name of facebook account to search
        account_id (int) : id of facebook account to search
    
    Returns:
        dict of data from result
    """

    try:
        # Make request of graph api
        result = request_with_retry(request_url)

        # If successful, parse result
        if result.status_code == 200:

            # Parse result
            result_json = result.json()

            return result_json['data']
        
        # If not successful, show result
        else:

            # Extract error message
            result_json = json.loads(result.text)
            error_message = result_json['error']['message']
            error_messsage_replaced = error_message.replace("'",'"').replace("`", '"').replace("-","+").replace("_","+")

            # Format account name for insertion
            account_name = "".join(ch for ch in account_name if ch.isalnum())
            account_name = account_name.replace("'", "").replace('"', "").replace("`","")

            # Store error in table for future checking
            insert_query = f'''INSERT INTO schoolclosure_adl.facebook_error VALUES (
                {account_id},
                '{account_name}',
                CURRENT_TIMESTAMP(),
                '{error_messsage_replaced}'
            )
            '''
            spark.sql(insert_query)

            return
        
    except Exception as e:
        raise e

# COMMAND ----------

def make_graph_request(account_id: str, account_name: str, since_timestamp: str, token: str) -> list:
    """Submit graph request for given account_id and time, return data from request.FacebookAnalyze

    Args:
        account_id (string) : facebok account id to query
        account_name (string) : name of facebook account id to query
        since_timestamp (string) : earliest time to query posts from, can be string date or unix timestamp
        token (string) : facebook access token

    Returns:
        list of data dictionaries from result
    """

    # Create formatted request url
    request_url = build_graph_api_url(account_name, since_timestamp, token)

    # Submit request
    request_data = graph_api_get(request_url, account_name, account_id)

    return request_data

# COMMAND ----------

def get_most_recent_post_info(request_data: list) -> str:
    """Extract most recent post id and timestamp from json data returned by request

    Args:
        request_data (list) - dictionary of json data from graph api request

    Returns:
        tuple containing:
            string of most recent post_id retrieved
            string of most recent created_time retrieved
    """

    # If no results, return None
    if not request_data:
        return None
    if len(request_data) == 0:
        return None

    # Sort list by created_time
    sorted_list = sorted(request_data, key=lambda i: i['created_time'], reverse=True)

    # Get most recent post
    most_recent_post = sorted_list[0]

    # Get most recent post id
    most_recent_post_id = most_recent_post['id']

    # Get most recent post created_time
    most_recent_post_created_time = most_recent_post['created_time']

    return (most_recent_post_id, most_recent_post_created_time)

# COMMAND ----------

def get_new_posts(account_id: str, account_name: str, since_timestamp: str, token: str) -> list:
    """Submit graph request for given account_id and time, return data from request.FacebookAnalyze

    Args:
        account_id (string) : facebok account id to query
        account_name (string) : facebook account name to query
        since_timestamp (string) : earliest time to query posts from, can be string date or unix timestamp
        token (string) : facebook access token

    Returns:
        tuple containing:
            string of most recent post_id retrieved
            string of most recent created_time retrieved
            list of data dictionaries from result
    """

    # Get new posts
    request_data = make_graph_request(account_id, account_name, since_timestamp, token)

    # Check if new posts exist
    if request_data:
        if len(request_data) > 0:

            # Get most recent post info
            new_most_recent_post_info = get_most_recent_post_info(request_data)

            # Return most recent post id
            return_tuple = (new_most_recent_post_info[0], new_most_recent_post_info[1], request_data)

            return return_tuple
        
        # If no new posts exist, return tuple with Nones
        else:
            return (None, None, None)
    # If no new posts exist, return tuple with Nones
    else:
        return (None, None, None)

# COMMAND ----------

def unpack_attachments(attachments_object: dict):
    """Breaks apart the attachments object returned by the Graph API into usable text for storage in database.
    
    Args:
        attachments_object (dict) - all info for attachments on a given post

    Returns:
        tuple:
            attachment_text (list) list of text strings
            attachment_img_links (list) list of image url strings
            attachment_targets (list) list of target dicts
    """
    # Establish lists for various retrieved objects
    attachment_text = []
    attachment_img_links = []
    attachment_targets = []

    # If there is no data, return empty lists
    if isinstance(attachments_object, float):
        return attachment_text, attachment_img_links, attachment_targets

    # Get list of data elements
    data_list = attachments_object['data']

    # If there is no data, return empty lists
    if not isinstance(attachments_object['data'], list):
        return attachment_text, attachment_img_links, attachment_targets

    # Iterate through data elements, extract text and image links
    for item in data_list:

        # Get item text
        description = item.get('description')
        if isinstance(description, str):
            attachment_text.append(str(description))

        # Get media
        media = item.get('media')
        if media:
            # Get image
            image = media.get('image')
            if image:
                # Get image source link
                image_src = image.get('src')
                if image_src:
                    attachment_img_links.append(image_src)

            # Get title
            title = media.get('title')
            if isinstance(title, str):
                attachment_text.append(str(title))

            # Get url
            url = media.get('url')
            if url:
                print("URL: ", url)
                attachment_img_links.append(url)

        # Get target
        target = item.get('target')
        if target:
            attachment_targets.append(target)

    # Remove duplicates
    attachment_text = list(set(attachment_text))
    attachment_img_links = list(attachment_img_links)
    attachment_targets = list(map(dict, set(tuple(sorted(d.items())) for d in attachment_targets)))

    return attachment_text, attachment_img_links, attachment_targets

# COMMAND ----------

def compile_fulltext(message: str, attachment_text: list) -> str:
    """Combine main message with attachment text, deduplicating to ensure there are no redundant posts.FacebookAnalyze

    Args:
        message (str) - initial message from facebook post
        attachment_text (list) - list of strings from attachment
    """

    if not attachment_text:
        return message

    if len(attachment_text) >= 0:
        # Ensure attachment text is a list
        attachment_text = [x for x in list(attachment_text) if isinstance(x, str)]

        # Add message to list and deduplicate
        attachment_text.insert(0, message)
        attachment_text = list(set(attachment_text))

        # Ensure all values are strings
        attachment_text = [i for i in attachment_text if isinstance(i, str)]

        # Join into long string
        final_string = ' '.join(attachment_text)

        return final_string
    
    else:
        return message

# COMMAND ----------

def generate_posts_df(post_data: list, account_id: str):
    """Take new posts and generate pandas dataframe.

    Args:
        post_data (list) - list of dictionaries
        account_id (str) - account_id from schoolclosure.facebook_accounts to be mapped to each post

    Returns:
        post_df - pandas dataFrame of posts
    """

    # Create DataFrame of posts
    post_df = pd.DataFrame(post_data)

    # Ensure attachments column exists
    if 'attachments' in post_df:
        # Unpack attachment data
        post_df['attachment_text'], post_df['attachment_img'], post_df['attachment_targets'] = zip(*post_df['attachments'].apply(lambda x: unpack_attachments(x)))

        # Convert type of each tuple to list
        post_df['attachment_img'] = post_df['attachment_img'].apply(lambda x: list(x))
        post_df['attachment_targets'] = post_df['attachment_targets'].apply(lambda x: list(x))

    else:
        post_df['attachments'] = None
        post_df['attachment_text'] = None
        post_df['attachment_img'] = None
        post_df['attachment_targets'] = None

    # Ensure message column exists
    if 'message' not in post_df:
        post_df['message'] = None

    # Compile full text
    post_df['full_text'] = post_df.apply(lambda x: compile_fulltext(x['message'], x['attachment_text']), axis=1)

    # Add account id
    post_df['facebook_account_id'] = int(account_id)

    # Convert created_time to datetime object
    post_df['created_time'] = pd.to_datetime(post_df['created_time'])

    return post_df

# COMMAND ----------

def get_post_df(post_data: list, account_id: str):
    """Take new posts and create spark dataframe which can be exported to transient table for merge.

    Args:
        post_data (list) - list of dictionaries
        account_id (str) - account_id from schoolclosure.facebook_accounts to be mapped to each post

    Returns:
        spark DataFrame with formatted posts
    """

    # Generate pandas dataframe
    post_df_pd = generate_posts_df(post_data, account_id)

    # Spark schemas
    attachments_schema = StructType([
        StructField('data', ArrayType(StructType([
            StructField('description', StringType(), True),
            StructField('media', StructType([
                StructField('image', StructType([
                    StructField('height', LongType(), True),
                    StructField('src', StringType(), True),
                    StructField('width', LongType(), True)
                    ]), True)
                ]), True),
            StructField('target', StructType([
                StructField('id', StringType(), True),
                StructField('url', StringType(), True)
            ])
            , True),
            StructField('url', StringType(), True)
        ])), True)
    ])

    attachments_targets_schema = ArrayType(
            StructType([
                StructField('id', StringType(), True),
                StructField('url', StringType(), True)
            ])      
        )
    
    post_df_schema = StructType([
        StructField('created_time', TimestampType(), False),
        StructField('id', StringType(), False),
        StructField('message', StringType(), True),
        StructField('permalink_url', StringType(), True),
        StructField('attachments', attachments_schema, True),
        StructField('attachments_text', ArrayType(StringType()), True),
        StructField('attachment_img', ArrayType(StringType()), True),
        StructField('attachment_targets', attachments_targets_schema, True),
        StructField('full_text', StringType(), True),
        StructField('facebook_account_id', IntegerType(), False)
        ]
    )

    # Order columns from pandas dataframe properly
    col_order = ['created_time', 'id', 'message', 'permalink_url',
                 'attachments', 'attachment_text', 'attachment_img',
                 'attachment_targets', 'full_text', 'facebook_account_id']
    post_df_pd = post_df_pd[col_order]

    # Create spark dataframe
    try:
        # Replace nulls with None for spark conversion
        post_df_pd = post_df_pd.where(cond=post_df_pd.notna(), other=None)
        post_df_spark = spark.createDataFrame(post_df_pd, schema = post_df_schema)
    except Exception as e:
        print("SPARK EXCEPTION: ", e)
        print("POST DF: ")
        print("LEN: ", len(post_df_pd))
        print(post_df_pd.info())
        raise e

    return post_df_spark

# COMMAND ----------

def run_get_new_posts(account_id: str, fb_account_id: str, since_timestamp: str, current_dt: str, token: str, verbose: bool = False):
    """Run process of getting new posts for a given facebook account. Store new data if found.

    Args:
        account_id (string) : account id from facebook_accounts table
        fb_account_id (string) : facebook account id to search
        since_timestamp (string) : earliest time to query posts from, can be string date or unix timestamp
        current_dt (string) : string of current timestamp
        token (string) : facebook access token
        verbose (boolean) : flag to print for debugging, default is False
    """
    if verbose:
        print("----------------------------------")
        print(f"RUNNING PROCESS FOR ACCOUNT ID {fb_account_id}")

    # Get new posts given account information
    try:
        if verbose:
            print(f"GETTING NEW POSTS FROM GRAPH API")
        returned_info = get_new_posts(account_id, fb_account_id, since_timestamp, token)
        last_post_id = returned_info[0]
        last_posted_at = returned_info[1]
        post_data = returned_info[2]
    except Exception as e:
        if verbose:
            print(f"ERROR WITH RETURNING POSTS FROM FACEBOOK: {e}")
        return

    # Check if new posts were returned
    if last_post_id is not None:

        if verbose:
            print("NEW POSTS FOUND")
        # Format post data for table upload
        posts_spark_df = get_post_df(post_data, account_id)

        if verbose:
            print("WRITING POSTS TO facebook_posts_merge TABLE")

        # Write to temp table for merge
        posts_spark_df.write.saveAsTable('schoolclosure_adl.facebook_posts_merge', mode='append')

        return

    else:
        if verbose:
            print("NO NEW POSTS FOUND")

        return

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute Ingest

# COMMAND ----------

# Set testing parameter
testing = False

if testing:
    record_limit = 1000
    accounts_pd = accounts_pd.head(record_limit)
    accounts_df = accounts_df.limit(record_limit)
    verbose = True
else:
    verbose = False  

# COMMAND ----------

if env=='dev':
  accounts_pd.head()

# COMMAND ----------

# MAGIC %sql
# MAGIC TRUNCATE TABLE schoolclosure_adl.facebook_posts_merge;

# COMMAND ----------

process_df = accounts_df.filter('account_error==False')

# COMMAND ----------

if env=='dev':
    process_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Graph API url for request

# COMMAND ----------

def build_graph_api_url(account_id: str, since_timestamp: str, token: str) -> str:
    """Generate url to hit with Facebook Graph API request.

    Args:
        account_id (string) : facebok account id to query
        since_timestamp (string) : earliest time to query posts from, can be string date or unix timestamp
        token (string) : facebook access token

    Returns:
        graph_url (string)
    """

    # Fields to capture
    request_fields = 'created_time,id,message,permalink_url,attachments{description,media,target,title,url}'

    # Handle null timestamp
    if isinstance(since_timestamp, pd.Timestamp):
        # Convert to unix
        since_timestamp = since_timestamp.timestamp()

        # Build request
        request_url = f'https://graph.facebook.com/v18.0/{account_id}/feed?fields={request_fields}&since={since_timestamp}&access_token={token}'

    else:
        # Build request
        request_url = f'https://graph.facebook.com/v18.0/{account_id}/feed?fields={request_fields}&access_token={token}'

    return request_url

# Build graph api url for call
udf_build_graph_api_url = f.udf(build_graph_api_url, StringType())

# COMMAND ----------

# Store token as column for building url
process_df = process_df.withColumn('token', f.lit(facebook_token))

# COMMAND ----------

# Iterate over rows, consolidate graph api url
process_df = process_df.withColumn('graph_api_url', udf_build_graph_api_url('fb_account_id', 'last_posted_at', 'token'))

# COMMAND ----------

if env=='dev':
    process_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Make Graph API call

# COMMAND ----------

def graph_api_get(request_args: dict) -> dict:
    """Get results from Facebook Graph API

    Args:
        request_args (dict) : dict of arguments for requesting info from graph api
    
    Returns:
        dict of results
    """

    request_url = request_args.get('graph_api_url')
    account_id = request_args.get('id')
    account_name = request_args.get('fb_account_id')
    request_num = request_args.get('request_num')

    return_dict = {}

    # Print every 100 requests
    if request_num % 100 == 0:
        print(f"PROCESSING REQUEST {request_num}")

    try:
        result = requests.get(request_url, verify=False)

        # Parse result
        result_json = result.json()

        return_dict['status_code'] = result.status_code
        return_dict['result_json'] = result_json
        return_dict['id'] = account_id
        return_dict['account_name'] = account_name

        return return_dict
    
    # Handling ill formed JSON in return
    except json.JSONDecodeError as e:
        return_dict['status_code'] = 999
        return_dict['result_json'] = {"error": "JSONDecodeError",
                                      "message": "JSONDecodeError"}
        return_dict['id'] = account_id
        return_dict['account_name'] = account_name

        return return_dict
        
    except Exception as e:
        raise e

def async_graph_api_get(request_dicts: list, threads: int):

    # Establish thread pool
    pool = ThreadPool(threads)

    # Run requests concurrently, retrieve results
    results = [pool.apply_async(graph_api_get, args=(request_dict,)) for request_dict in request_dicts]

    # Close out pool
    pool.close()
    pool.join()

    # Get all async results
    return_results = [x.get() for x in results]

    return return_results

# COMMAND ----------

# Consolidate list of dicts with query args
dict_df = process_df.select('id', 'graph_api_url', 'fb_account_id')

# Create row number
w = Window().orderBy(f.col('id'))
dict_df = dict_df.withColumn("request_num", f.row_number().over(w))

# Map to dicts for proessing
query_dicts = dict_df.rdd.map(lambda row: row.asDict()).collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest & Processing Loop

# COMMAND ----------

# MAGIC %md
# MAGIC ### Process Graph API results

# COMMAND ----------

# MAGIC %md
# MAGIC #### Handle Errored Requests

# COMMAND ----------

def handle_errors(results):
    success_results = []
    error_results = []
    for r in results:
        if r.get('status_code')==200:
            success_results.append(r)
        else:
            error_results.append(r)

    print(f"SUCCESSFULLY PROCESSED {len(success_results)} ACCOUNTS")
    print(f"ERRORS WITH {len(error_results)} ACCOUNTS")


    for error in tqdm(error_results):

        account_id = error.get('id')
        account_name = error.get('account_name')
        result_json = error.get('result_json')
        error_dict = result_json.get('error')
        error_message = error_dict.get('message')
        error_message = error_message.replace("'",'"').replace("`", '"').replace("-","+").replace("_","+")


        # Store error in table for future checking
        insert_query = f'''INSERT INTO schoolclosure_adl.facebook_error VALUES (
            {account_id},
            '{account_name}',
            CURRENT_TIMESTAMP(),
            '{error_message}'
        )
        '''
        spark.sql(insert_query)

    return success_results

# COMMAND ----------

# MAGIC %md
# MAGIC ### Store Successful Results

# COMMAND ----------

def unpack_attachments(attachments_object: dict):
    """Breaks apart the attachments object returned by the Graph API into usable text for storage in database.
    
    Args:
        attachments_object (dict) - all info for attachments on a given post

    Returns:
        tuple:
            attachment_text (list) list of text strings
            attachment_img_links (list) list of image url strings
            attachment_targets (list) list of target dicts
    """
    # Establish lists for various retrieved objects
    attachment_text = []
    attachment_img_links = []
    attachment_targets = []

    # Convert to dict
    if attachments_object is not None:
        attachments_object = attachments_object.asDict(True)

    # If there is no data, return empty lists
    if not attachments_object:
        return {'attachment_text': attachment_text, 
            'attachment_img_links': attachment_img_links, 
            'attachment_targets': attachment_targets,
            'attachment_object': str(attachments_object)}
    if isinstance(attachments_object, float):
        return {'attachment_text': attachment_text, 
            'attachment_img_links': attachment_img_links, 
            'attachment_targets': attachment_targets,
            'attachment_object': str(attachments_object)}

    # Get list of data elements
    data_list = attachments_object.get('data', [])

    # If there is no data, return empty lists
    if not isinstance(data_list, list):
        return {'attachment_text': attachment_text, 
            'attachment_img_links': attachment_img_links, 
            'attachment_targets': attachment_targets,
            'attachment_object': str(attachments_object)}

    # If there is no data, return empty lists
    if len(data_list)==0:
        return {'attachment_text': attachment_text, 
            'attachment_img_links': attachment_img_links, 
            'attachment_targets': attachment_targets,
            'attachment_object': str(attachments_object)}

    # Iterate through data elements, extract text and image links
    for item in data_list:

        if isinstance(item, dict):

            # Get item text
            description = item.get('description', None)
            if isinstance(description, str):
                attachment_text.append(str(description))

            # Get media
            media = item.get('media', None)
            if media:
                # Get image
                image = media.get('image', None)
                if image:
                    # Get image source link
                    image_src = image.get('src', None)
                    if image_src:
                        attachment_img_links.append(image_src)

                # Get title
                title = media.get('title', None)
                if isinstance(title, str):
                    attachment_text.append(str(title))

                # Get url
                url = media.get('url', None)
                if url:
                    print("URL: ", url)
                    attachment_img_links.append(url)

            # Get target
            target = item.get('target', None)
            if target:
                attachment_targets.append(target)

    # Remove duplicates
    attachment_text = list(set(attachment_text))
    attachment_img_links = list(attachment_img_links)
    attachment_targets = list(map(dict, set(tuple(sorted(d.items())) for d in attachment_targets)))

    return {'attachment_text': attachment_text, 
            'attachment_img_links': attachment_img_links, 
            'attachment_targets': attachment_targets,
            'attachment_object': str(attachments_object)}
    
attachments_targets_schema = ArrayType(
        StructType([
            StructField('id', StringType(), True),
            StructField('url', StringType(), True)
        ])      
    )

attachments_parsed_schema = StructType([
    StructField('attachment_text', ArrayType(StringType()), True),
    StructField('attachment_img_links', ArrayType(StringType()), True),
    StructField('attachment_targets', attachments_targets_schema, True),
    StructField('attachment_object', StringType(), True)
])

udf_unpack_attachments = f.udf(unpack_attachments, attachments_parsed_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Compile Fulltext

# COMMAND ----------

def compile_fulltext(message: str, attachment_text: list) -> str:
    """Combine main message with attachment text, deduplicating to ensure there are no redundant posts.FacebookAnalyze

    Args:
        message (str) - initial message from facebook post
        attachment_text (list) - list of strings from attachment
    """

    if not attachment_text:
        return message

    if len(attachment_text) >= 0:
        # Ensure attachment text is a list
        attachment_text = [x for x in list(attachment_text) if isinstance(x, str)]

        # Add message to list and deduplicate
        attachment_text.insert(0, message)
        attachment_text = list(set(attachment_text))

        # Ensure all values are strings
        attachment_text = [i for i in attachment_text if isinstance(i, str)]

        # Join into long string
        final_string = ' '.join(attachment_text)

        return final_string
    
    else:
        return message
    
udf_compile_fulltext = f.udf(compile_fulltext, StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compile Final DataFrame

# COMMAND ----------

def format_dataframe(success_results):
    # Spark schemas
    attachments_schema = StructType([
        StructField('data', ArrayType(StructType([
            StructField('description', StringType(), True),
            StructField('media', StructType([
                StructField('image', StructType([
                    StructField('height', LongType(), True),
                    StructField('src', StringType(), True),
                    StructField('width', LongType(), True)
                    ]), True)
                ]), True),
            StructField('target', StructType([
                StructField('id', StringType(), True),
                StructField('url', StringType(), True)
            ])
            , True),
            StructField('url', StringType(), True)
        ])), True)
    ])

    attachments_targets_schema = ArrayType(
            StructType([
                StructField('id', StringType(), True),
                StructField('url', StringType(), True)
            ])      
        )

    result_json_schema = StructType([
        StructField('data', ArrayType(StructType([
            StructField('id', StringType(), False),
            StructField('created_time', StringType(), False),
            StructField('message', StringType(), True),
            StructField('permalink_url', StringType(), True),
            StructField('attachments', attachments_schema, True)
    ])), True)
    ])

    post_df_schema = StructType([
        StructField('id', IntegerType(), False),
        StructField('account_name', StringType(), False),
        StructField('status_code', IntegerType(), False),
        StructField('result_json', result_json_schema, False)
        ]
    )

    result_df = spark.createDataFrame(success_results, post_df_schema)

    result_df = result_df.withColumnRenamed('id', 'facebook_account_id')

    result_df = result_df.withColumn('data', f.col('result_json')['data']) \
                     .withColumn('post', f.explode(f.col('data'))) \
                     .withColumn('id', f.col('post')['id']) \
                     .withColumn('created_time', f.col('post')['created_time']) \
                     .withColumn('message', f.col('post')['message']) \
                     .withColumn('permalink_url', f.col('post')['permalink_url']) \
                     .withColumn('attachments', f.col('post')['attachments']) \
                     .select('facebook_account_id', 'created_time', 'account_name', 'id', 'message', 'permalink_url', 'attachments')

    # Unpack all attachments
    result_df = result_df.withColumn('attachments_unpacked', udf_unpack_attachments(f.col('attachments'))) \
                        .withColumn('attachments_text', f.col('attachments_unpacked')['attachment_text']) \
                        .withColumn('attachment_img', f.col('attachments_unpacked')['attachment_img_links']) \
                        .withColumn('attachment_targets', f.col('attachments_unpacked')['attachment_targets'])
    result_df.cache().count()

    result_df = result_df.withColumn('full_text', udf_compile_fulltext('message', 'attachments_text'))

    post_df = result_df.withColumn('created_time', f.to_timestamp("created_time", "yyyy-MM-dd'T'HH:mm:ss+SSSS"))

    # Select final columns
    post_df = post_df.select('created_time', 'id', 'message', 'permalink_url', 'attachments',
                            'attachments_text', 'attachment_img', 'attachment_targets', 'full_text', 'facebook_account_id')

    return post_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write / Merge

# COMMAND ----------

def merge_dataframe(post_df):

    post_df.write.saveAsTable('schoolclosure_adl.facebook_posts_merge', mode='overwrite')

    spark.sql('''
    MERGE INTO schoolclosure_adl.facebook_posts dest
     USING (SELECT DISTINCT * FROM schoolclosure_adl.facebook_posts_merge) source
        ON (dest.id=source.id)
      WHEN NOT MATCHED
      THEN INSERT (created_time, id, message, permalink_url, attachments, attachments_text, attachment_img, attachment_targets,
                   full_text, facebook_account_id)
           VALUES (created_time, id, message, permalink_url, attachments, attachments_text, attachment_img, attachment_targets,
                   full_text, facebook_account_id);          
              ''')
    
    return

# COMMAND ----------

# MAGIC %md
# MAGIC ### Process Accounts

# COMMAND ----------

def update_accounts(post_df):
    # Get dataframe of all accounts processed
    accounts_processed = post_df.select('facebook_account_id') \
                                .withColumnRenamed('facebook_account_id', 'id') \
                                .dropDuplicates()

    # Create table of processed accounts
    accounts_processed.createOrReplaceTempView('facebook_accounts_processed')                       

    # Note accounts to update
    spark.sql('''
              CREATE OR REPLACE TEMPORARY VIEW facebook_accounts_merge AS (
    WITH all_posts AS (
        SELECT fpm.facebook_account_id,
            fpm.created_time,
            fpm.id,
            ROW_NUMBER() OVER (PARTITION BY fpm.facebook_account_id ORDER BY fpm.created_time DESC) AS ROW_NUM
        FROM schoolclosure_adl.facebook_posts_merge fpm
    ),
        most_recent_posts AS (
        SELECT *
            FROM all_posts
        WHERE ROW_NUM = 1
        )

    SELECT fap.id AS ID,
            mrp.id AS last_post_id,
            mrp.created_time AS last_posted_at
        FROM facebook_accounts_processed fap
        FULL OUTER JOIN most_recent_posts mrp
        ON fap.id = mrp.facebook_account_id
    );
              ''')
    
    # Update facebook Accounts
    spark.sql('''
        MERGE INTO schoolclosure_adl.facebook_accounts dest
     USING (SELECT * FROM facebook_accounts_merge) source
        ON (dest.id=source.id)
      WHEN MATCHED AND source.last_post_id IS NOT NULL
      THEN UPDATE
       SET dest.last_post_id = source.last_post_id,
           dest.last_posted_at = source.last_posted_at;      

              ''')
    
    return

# COMMAND ----------

def ingest_process_loop(query_dicts, chunk_id, chunk_length):

    # Get results from facebook api
    print(f"QUERYING FACEBOOK API: {chunk_id+1}/{chunk_length}")
    results = async_graph_api_get(query_dicts, threads=8)

    # First, handle errored results
    success_results = handle_errors(results)

    # Format result dataframe
    post_df = format_dataframe(success_results)

    # Merge into merge table
    print(f"STORING POSTS: {chunk_id+1}/{chunk_length}")
    merge_dataframe(post_df)

    # Update accounts
    print(f"UDPATING ACCOUNTS: {chunk_id+1}/{chunk_length}")
    update_accounts(post_df)

    return

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loop and Ingest

# COMMAND ----------

accounts_per_chunk = 5000

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Split query_dicts into chunks of smaller size
query_dicts_chunks = [x for x in chunks(query_dicts, accounts_per_chunk)]
print("NUM CHUNKS: ", len(query_dicts_chunks))

# COMMAND ----------

chunks_to_process = query_dicts_chunks

for idx, chunk in enumerate(chunks_to_process):
    print(f"RUNNING CHUNK {idx+1}/{len(chunks_to_process)}")
    ingest_process_loop(chunk, idx, len(chunks_to_process))

# COMMAND ----------

# Exit notebook with success message
dbutils.notebook.exit('SUCCESS')
