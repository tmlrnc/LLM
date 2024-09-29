
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline
from langchain.chains import RetrievalQA

# Part I: LSTM Model for Health Risk Prediction

# Sample wearable data (heart rate, glucose levels, activity)
data = np.array([[72, 130, 5], [85, 200, 2], [60, 95, 10], [90, 210, 1], [70, 105, 6]])
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Define LSTM Model for Time-Series Prediction
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Prepare Data for LSTM
input_size = 3  # heart rate, glucose, activity
hidden_size = 50
output_size = 1  # health risk prediction (0 = low risk, 1 = high risk)

lstm_model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X = torch.tensor(data_normalized, dtype=torch.float32).unsqueeze(1)
y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32).unsqueeze(1)

# Train the LSTM model
for epoch in range(100):
    lstm_model.train()
    optimizer.zero_grad()
    outputs = lstm_model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Test the LSTM model with new data
test_data = np.array([[80, 150, 3]])  # New data (heart rate, glucose, activity)
test_data_normalized = scaler.transform(test_data)
X_test = torch.tensor(test_data_normalized, dtype=torch.float32).unsqueeze(1)
lstm_model.eval()
prediction = lstm_model(X_test)
print(f"Predicted health risk: {prediction.item()}")

# Part II: Retrieval-Augmented Generation (RAG) with LangChain & GPT for Advice Generation

# Simulate a user symptom input for GPT-based advice generation
user_symptom = "I feel dizzy and my glucose levels are high."

# Pre-trained GPT model for conversational AI
qa_pipeline = pipeline("question-answering")

# Medical knowledge base (for demonstration)
medical_docs = """
Dizziness can be caused by various factors including high glucose levels.
It is important to monitor glucose levels regularly to avoid complications.
"""

# Use GPT model to generate advice based on symptoms
result = qa_pipeline(question=user_symptom, context=medical_docs)
print(f"GPT-based Advice: {result['answer']}")

# LangChain-based Retrieval with GPT Integration for dynamic document retrieval
retrieval_chain = RetrievalQA(model=qa_pipeline, documents=[medical_docs])
print(retrieval_chain.run(user_symptom))

# Part III: File-Based Query Retrieval using RAG (HR Document Example)

# Load pre-trained model for question-answering
def fetch_top_3_paragraphs_from_file(file_path, query):
    # Read the HR policy document
    with open(file_path, 'r') as file:
        document = file.read()
    
    # Split document into paragraphs
    paragraphs = document.split('\n\n')
    results = []
    
    # Iterate through paragraphs and perform QA model to find relevant sections
    for para in paragraphs:
        if para.strip():
            result = qa_pipeline(question=query, context=para)
            results.append((para, result['score']))
    
    # Sort paragraphs by the relevance score and return top 3
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return [para for para, score in results[:3]]

# Example usage with the provided HR document
file_path = "HR_Policy_Doc.txt"  # Path to the HR policy document
query = "What is the procedure for sexual harassment?"

top_3_paragraphs = fetch_top_3_paragraphs_from_file(file_path, query)
for i, para in enumerate(top_3_paragraphs, 1):
    print(f"Paragraph {i}:\n{para}\n")

# Integration for Health Assistant: This process can be adapted to retrieve medical documents or EHRs and provide advice.
