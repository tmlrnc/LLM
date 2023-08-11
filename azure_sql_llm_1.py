import pyspark
from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import openai
import re
import requests
import sys
import os
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast



# Report 1 - Denial rate by Cohort of Race Ethnicity and Gender

  
# creating the dataset
data = {'White':20, 'Black':35, 'Latino':33,'Asian':35,'Male':5, 'Female':30,'Transgender':45}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='green',
        width = 0.4)
 
plt.xlabel("COHORTS")
plt.ylabel("Denial Rate")
plt.title("VA Claim Denial Rate by Cohort - Race Ethnicity Gender")
plt.show()



## Report 2 -  Denial reasons by Cohort of Race Ethnicity and Gender

 
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
 
# set height of bar
IT = [12, 30, 1, 8, 22]
ECE = [28, 6, 16, 5, 10]
CSE = [29, 3, 24, 25, 17]
 
# Set position of bar on X axis
br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, IT, color ='r', width = barWidth,
        edgecolor ='grey', label ='NO INSURANCE')
plt.bar(br2, ECE, color ='g', width = barWidth,
        edgecolor ='grey', label ='NOT COVERED')
plt.bar(br3, CSE, color ='b', width = barWidth,
        edgecolor ='grey', label ='NOT NEEDED')
 
# Adding Xticks
plt.xlabel('Cohort', fontweight ='bold', fontsize = 15)
plt.ylabel('Denial Reason Percentage', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(IT))],
        ['White', 'Black', 'Latino', 'Asian', 'Native American'])
 


plt.legend()



# Report 3 - The total # claims by Cohort of Race Ethnicity and Gender


plt.title("Number of Claims by Cohort - Race Ethnicity Gender")


# libraries
import matplotlib.pyplot as plt
import numpy as np

# create dataset

height = [500, 400, 500,450,330, 310, 320]
  
# giving column names of dataframe - claim denial rate 
bars = ["White", "Black", "Latino","Asian", "Male", "Femail", "Transgender"]


y_pos = np.arange(len(bars))
 
# Create horizontal bars
plt.barh(y_pos, height)
 
# Create names on the x-axis
plt.yticks(y_pos, bars)
 
# Show graphic
plt.show()


exit()




#######################################
# Azure databricks SQL for reports 
#######################################








host = os.getenv("DATABRICKS_HOST")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
access_token = os.getenv("DATABRICKS_ACCESS_TOKEN")

connection = sql.connect(
  server_hostname=host,
  http_path=http_path,
  access_token=access_token)

cursor = connection.cursor()

sql_claim_grant_rate_simulated_data_by_cohort = "SELECT race, ethnicity,gender, COUNT(*) AS total_claims, AVG(CASE WHEN claim_status = 'Granted' THEN 1 ELSE 0 END) AS claim_grant_rate, AVG(CASE WHEN claim_status = 'Denied' THEN 1 ELSE 0 END) AS claim_denial_rate FROM claims GROUP BY race, ethnicity,gender;"


cursor.execute(sql_claim_grant_rate_simulated_data_by_cohort)
result = cursor.fetchall()
for row in result:
  claim_grant_rate_data_by_cohort = row








Claim_Grant_Reasons = ""

sql_claim_grants = "SELECT race,ethnicity,gender,reason, COUNT(*) AS count FROM claims WHERE claim_status = 'Granted' GROUP BY race, ethnicity, gender, reason ORDER BY count DESC LIMIT 10;"

cursor.execute(sql_claim_grants)
result = cursor.fetchall()
for row in result:
  Claim_Grant_Reasons = Claim_Grant_Reasons + row
  print(row)




#######################################
# Azure OPEN AI for customer service LLM generaive AI to reduce claim denial rate
#######################################




prompt=f"""
Follow exactly those 3 steps:
1. Read the Claim Grant reasons below and aggregate this data
Context : {Claim_Grant_Reasons}
2. Answer the question using only this context
3. Show the source for your answers
User Question: How do I get my Claim Granted?
"""


import openai
openai.api_type = "azure"
openai.api_key = config['OPENAI_API_KEY']
openai.api_base = "https://openaitest123.openai.azure.com/"
openai.api_version = "2022-12-01"

response = openai.Completion.create(
          engine="test1",
          prompt=prompt,
          temperature=0,
          max_tokens=1041,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          best_of=1,
          stop=None)


print(f"Question: \n{question}")
print(f"Response: \n{response.text}")




