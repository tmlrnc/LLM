import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, pipeline
from sklearn.metrics import accuracy_score
import optuna  # For Bayesian optimization


# Define a TrovoHealthUser class to represent individual users
class TrovoHealthUser:
    def __init__(self, username, model, tokenizer):
        self.username = username
        self.friends = []
        self.tweets = []
        self.model = model
        self.tokenizer = tokenizer

    def add_friend(self, friend_user):
        if friend_user not in self.friends:
            self.friends.append(friend_user)
        else:
            print(f"{friend_user.username} is already a friend.")

    def send_tweet(self, tweet_content):
        # Predict health-related risk if the tweet contains health information
        if "dizzy" in tweet_content or "fatigue" in tweet_content:
            sample_time_series = [0.25, 0.4, -0.1, 0.8, 0.9]  # Example time-series data
            predicted_risk, recommendation = predict_single_patient(self.model, self.tokenizer, tweet_content, sample_time_series)
            tweet_content += f"\nHealth Prediction: {predicted_risk}. Recommendation: {recommendation}"
        
        tweet = {'user': self.username, 'content': tweet_content}
        self.tweets.append(tweet)
        print(f"Tweet sent by {self.username}: {tweet_content}")

    def read_friends_tweets(self):
        tweets = []
        for friend in self.friends:
            tweets.extend(friend.tweets)
        return tweets

# Define the TrovoHealthTwitter class that manages multiple users and interactions
class TrovoHealthTwitter:
    def __init__(self, model, tokenizer):
        self.users = {}
        self.model = model
        self.tokenizer = tokenizer

    def add_user(self, username):
        if username not in self.users:
            self.users[username] = TrovoHealthUser(username, self.model, self.tokenizer)
        else:
            print(f"User {username} already exists.")

    def get_user(self, username):
        return self.users.get(username, None)

# Simulate data creation (re-used from your original code)
def simulate_data(num_samples=10):
    data = {
        'text': [f'Patient report {i} showing symptoms of fatigue and dizziness.' for i in range(num_samples)],
        'time_series': [torch.randn(5).tolist() for _ in range(num_samples)],  # Example time-series data
        'label': torch.randint(0, 2, (num_samples,)).tolist()  # Random binary labels (0: Low Risk, 1: High Risk)
    }
    return pd.DataFrame(data)

# Define a custom dataset
class HealthDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Tokenize text using BERT tokenizer
        inputs = self.tokenizer.encode_plus(
            row['text'],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Prepare time-series data and label
        time_series = torch.tensor(row['time_series'], dtype=torch.float32)
        label = torch.tensor(row['label'], dtype=torch.long)

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'time_series': time_series,
            'label': label
        }

# Define the BERT-LSTM model architecture
class BertLSTMHealthcareModel(nn.Module):
    def __init__(self, bert_model_name, lstm_hidden_size, num_labels):
        super(BertLSTMHealthcareModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size + lstm_hidden_size, num_labels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, time_series):
        # BERT encoding for text data
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token output from BERT

        # LSTM for time-series data
        time_series = time_series.unsqueeze(-1)  # Add input size dimension for LSTM
        _, (hn, _) = self.lstm(time_series)  # hn is the final hidden state from LSTM

        # Concatenate BERT and LSTM outputs
        combined = torch.cat((pooled_output, hn.squeeze(0)), dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits

# RAG (Retrieval-Augmented Generation) to retrieve relevant documents
def retrieve_relevant_documents(query):
    documents = {
        "doc1": "Treatment for fatigue includes sleep hygiene, stress management, and exercise.",
        "doc2": "Patients with dizziness should monitor blood pressure and hydration levels."
    }
    return documents.values()

# Health recommendations based on the predicted risk and retrieved documents
def health_recommendation(predicted_risk, query):
    recommendations = {
        "Low Risk": "Maintain a balanced diet, regular exercise, and proper hydration. Continue monitoring symptoms but no immediate medical intervention is required.",
        "High Risk": "Consult a healthcare provider for immediate evaluation. Consider medical tests for underlying conditions and follow prescribed treatment for symptoms like fatigue and dizziness."
    }

    # Retrieve relevant documents using RAG based on the query
    docs = retrieve_relevant_documents(query)

    # Use GPT-2 or other generation models to augment the recommendation
    gpt = pipeline('text-generation', model="gpt-2")
    concatenated_docs = " ".join(docs)
    augmented_response = gpt(f"Patient risk level: {predicted_risk}. {recommendations[predicted_risk]} Relevant guidelines: {concatenated_docs}", max_length=150)
    
    return augmented_response[0]['generated_text']

# Function to predict for a single patient
def predict_single_patient(model, tokenizer, text, time_series_data, max_len=128):
    model.eval()

    # Tokenize the input text using BERT tokenizer
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    time_series = torch.tensor(time_series_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension and input size dimension

    # Run the model to predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, time_series=time_series)
        prediction = torch.argmax(outputs, dim=1)

    # Map the prediction to the actual risk class
    risk_mapping = {0: "Low Risk", 1: "High Risk"}
    predicted_risk = risk_mapping[prediction.item()]

    # Use RAG to retrieve relevant health information and augment the recommendation
    recommendation = health_recommendation(predicted_risk, text)

    return predicted_risk, recommendation

# Main function
def main():
    # Simulate data
    df = simulate_data(num_samples=20)

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Initialize the model
    model = BertLSTMHealthcareModel('bert-base-uncased', lstm_hidden_size=64, num_labels=2)
    model.eval()

    # Initialize TrovoHealthTwitter with the model and tokenizer
    trovo_twitter = TrovoHealthTwitter(model, tokenizer)

    # Add users
    trovo_twitter.add_user("Alice")
    trovo_twitter.add_user("Bob")

    # Get user objects
    alice = trovo_twitter.get_user("Alice")
    bob = trovo_twitter.get_user("Bob")

    # Add friends
    alice.add_friend(bob)

    # Send health-related tweets
    alice.send_tweet("Feeling a bit dizzy today, will see the doctor.")
    bob.send_tweet("Had a great workout session this morning!")

    # Read friends' tweets and print
    print("Alice's friends' tweets:", alice.read_friends_tweets())
    print("Bob's friends' tweets:", bob.read_friends_tweets())

if __name__ == "__main__":
    main()
