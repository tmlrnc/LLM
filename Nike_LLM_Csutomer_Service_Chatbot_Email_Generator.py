import openai
import pinecone
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Step 1: Set up Pinecone
pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='us-west1-gcp')
index_name_support = 'nike-customer-support'
pinecone.create_index(index_name_support, dimension=768, metric='cosine')
support_index = pinecone.Index(index_name_support)

# Step 2: Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate BERT embeddings
def generate_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings.flatten()

# Step 3: Store historical customer support data in Pinecone
historical_support_data = [
    {"question": "How can I return my shoes?", "answer": "You can return shoes within 30 days with proof of purchase."},
    {"question": "Where is my order?", "answer": "Track your order by logging into your account."},
]

for data in historical_support_data:
    text = f"Question: {data['question']} Answer: {data['answer']}"
    embedding = generate_bert_embedding(text)
    support_index.upsert(vectors=[(data['question'], embedding, {'answer': data['answer']})])

# Step 4: Set up LangChain for response generation
template = """Customer Question: {question}
Relevant Past Support: {support_data}
Generate a response to assist the customer."""

prompt = PromptTemplate(input_variables=["question", "support_data"], template=template)
llm = OpenAI(api_key='YOUR_OPENAI_API_KEY', model='text-davinci-003')
langchain = LLMChain(llm=llm, prompt=prompt)

# Step 5: Nike Customer Support Chatbot function
def nike_customer_support_chatbot(query):
    query_embedding = generate_bert_embedding(query)
    results = support_index.query(vector=query_embedding, top_k=5)
    retrieved_data = [match['metadata']['answer'] for match in results['matches']]
    combined_support_data = ' '.join(retrieved_data)
    response = langchain.run({"question": query, "support_data": combined_support_data})
    return response

# Step 6: Example of usage
customer_query = "How do I return my shoes?"
response = nike_customer_support_chatbot(customer_query)
print("Customer Support Chatbot Response:", response)

import openai
import pinecone
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
import pandas as pd

# Step 1: Initialize OpenAI, Pinecone, and Hugging Face models
openai.api_key = 'YOUR_OPENAI_API_KEY'
pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='us-west1-gcp')

# Create indexes for customer support and email subject line customization
index_name_support = 'nike-customer-support'
index_name_email = 'nike-email-subject-line-customization'
pinecone.create_index(index_name_support, dimension=768, metric='cosine')
pinecone.create_index(index_name_email, dimension=768, metric='cosine')

support_index = pinecone.Index(index_name_support)
email_index = pinecone.Index(index_name_email)

# Load BERT tokenizer and model from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Step 2: Function to generate BERT embeddings for a given text
def generate_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings.flatten()

# Step 3: Store historical customer support and email data in Pinecone
historical_support_data = [
    {"question": "How can I return my shoes?", "answer": "You can return shoes within 30 days with proof of purchase."},
    {"question": "Where is my order?", "answer": "Track your order by logging into your account."},
]

historical_email_data = [
    {"sales": "120 units sold", "demographic": "John Doe, New York", "subject_line": "Exclusive Offer on Nike Shoes"},
    {"sales": "200 units sold", "demographic": "Jane Smith, Los Angeles", "subject_line": "20% Off on Running Shoes"},
]

# Store data in the customer support index
for data in historical_support_data:
    text = f"Question: {data['question']} Answer: {data['answer']}"
    embedding = generate_bert_embedding(text)
    support_index.upsert(vectors=[(data['question'], embedding, {'answer': data['answer']})])

# Store data in the email subject line customization index
for data in historical_email_data:
    text = f"Sales: {data['sales']} Demographic: {data['demographic']} Subject Line: {data['subject_line']}"
    embedding = generate_bert_embedding(text)
    email_index.upsert(vectors=[(data['demographic'], embedding, {'subject_line': data['subject_line']})])

# Step 4: LangChain setup for customer support
template_support = """Customer Question: {question}
Relevant Past Support: {support_data}
Generate a response to assist the customer."""

prompt_support = PromptTemplate(
    input_variables=["question", "support_data"],
    template=template_support
)
llm_support = OpenAI(api_key='YOUR_OPENAI_API_KEY', model='text-davinci-003')
langchain_support = LLMChain(llm=llm_support, prompt=prompt_support)

# Step 5: LangChain setup for email subject line generation
template_email = """Customer Demographic: {demographic}
Sales Data: {sales_data}
Generate a personalized email subject line based on historical sales and interactions."""

prompt_email = PromptTemplate(
    input_variables=["demographic", "sales_data"],
    template=template_email
)
llm_email = OpenAI(api_key='YOUR_OPENAI_API_KEY', model='text-davinci-003')
langchain_email = LLMChain(llm=llm_email, prompt=prompt_email)

# Step 6: Function for Customer Support Chatbot
def nike_customer_support_chatbot(query):
    # Embed the customer query
    query_embedding = generate_bert_embedding(query)
    
    # Query Pinecone to retrieve relevant historical support data
    results = support_index.query(vector=query_embedding, top_k=5)
    retrieved_data = [match['metadata']['answer'] for match in results['matches']]
    
    # Combine retrieved data for GPT response
    combined_support_data = ' '.join(retrieved_data)
    
    # Generate a response using LangChain
    response = langchain_support.run({"question": query, "support_data": combined_support_data})
    return response

# Step 7: Function for Email Subject Line Generation
def nike_email_subject_line_generator(demographic, sales_data):
    # Embed the demographic and sales data
    customer_embedding = generate_bert_embedding(demographic + ' ' + sales_data)
    
    # Query Pinecone to retrieve relevant email subject lines
    results = email_index.query(vector=customer_embedding, top_k=5)
    retrieved_subject_lines = [match['metadata']['subject_line'] for match in results['matches']]
    
    # Combine retrieved data for GPT to generate a personalized subject line
    combined_subject_lines = ' '.join(retrieved_subject_lines)
    response = langchain_email.run({"demographic": demographic, "sales_data": combined_subject_lines})
    return response

# Step 8: Example usage of Customer Support Chatbot
customer_query = "How do I return my shoes?"
response_chatbot = nike_customer_support_chatbot(customer_query)
print("Customer Support Chatbot Response:", response_chatbot)

# Step 9: Example usage of Email Subject Line Generator
customer_demographic = "John Doe, New York"
customer_sales_data = "150 units sold, last purchased Nike Air Max"
response_subject_line = nike_email_subject_line_generator(customer_demographic, customer_sales_data)
print("Generated Email Subject Line:", response_subject_line)

