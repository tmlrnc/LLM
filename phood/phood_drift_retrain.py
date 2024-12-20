from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently import ColumnMapping
import pandas as pd

# Example: Reference dataset and new dataset
reference_data = pd.read_csv("reference_data.csv")
current_data = pd.read_csv("current_data.csv")

# Define column mapping (e.g., features and targets)
column_mapping = ColumnMapping(target=None, prediction=None)

# Create and run the report
drift_report = Report(metrics=[DataDriftTable()])
drift_report.run(reference_data=reference_data, current_data=current_data)

# Save the report
drift_report.save_html("data_drift_report.html")

# Interpret drift results
drift_results = drift_report.as_dict()
drifted_features = [
    feature["column_name"]
    for feature in drift_results["metrics"][0]["result"]["drift_by_columns"]
    if feature["drift_detected"]
]

print(f"Drifted Features: {drifted_features}")

from river.drift import ADWIN
import numpy as np

# Initialize drift detector
adwin = ADWIN()

# Simulated stream of feature data
feature_stream = np.random.normal(size=1000)  # Replace with actual stream

# Detect drift
for i, value in enumerate(feature_stream):
    adwin.update(value)
    if adwin.change_detected:
        print(f"Drift detected at index {i}!")


import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Load new data (e.g., images collected over 6 months)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

new_data = datasets.ImageFolder(root="path_to_new_data", transform=transform)
new_data_loader = DataLoader(new_data, batch_size=32, shuffle=True)

# Load pre-trained model
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features

# Adjust the final layer to match the new dataset
model.fc = nn.Linear(num_features, len(new_data.classes))

# Fine-tune the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(5):  # Train for a few epochs on new data
    epoch_loss = 0
    for inputs, labels in new_data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}")

 from river import linear_model
from river import optim
from river import preprocessing

# Online learning model
model = preprocessing.StandardScaler() | linear_model.LogisticRegression(optimizer=optim.SGD(0.01))

# Simulated data stream
data_stream = [
    ({"feature1": 0.1, "feature2": 0.3}, 1),
    ({"feature1": 0.4, "feature2": 0.2}, 0),
    # Add more examples as they arrive...
]

# Incrementally update model
for features, label in data_stream:
    model.learn_one(features, label)

# Make predictions
prediction = model.predict_one({"feature1": 0.2, "feature2": 0.4})
print(f"Prediction: {prediction}")