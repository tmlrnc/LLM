
"""
Environment Setup: You'll need to install the transformers, torch, and datasets libraries. This can be done via pip if they are not already installed.
Loading BERT Model and Tokenizer: The script loads a pre-trained BERT model (bert-base-uncased) and the corresponding tokenizer.
Dataset Preparation: The script uses the IMDb dataset as an example, tokenizing the text data and preparing it for training.
Training: The model is trained using the Trainer API provided by the transformers library. Training arguments like batch size, number of epochs, and evaluation strategy are defined.
Evaluation: After training, the model is evaluated on the test dataset.
Model Saving: The fine-tuned model and tokenizer are saved for later use.
Inference: Finally, the model is loaded again to perform inference on a sample sentence, and the predictions are printed.
"""

# Step 1: Set Up the Environment

# Install Required Libraries (to be run in your terminal, not in the Python script)
# !pip install transformers torch datasets

# Import the Libraries
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import torch

# Step 2: Load a Pre-trained BERT Model and Tokenizer

# Select a Pre-trained BERT Model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust `num_labels` for your task

# Step 3: Prepare the Dataset

# Load and Preprocess the Dataset
dataset = load_dataset("imdb")  # Example dataset; replace with your specific dataset

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Step 4: Set Up the Training Arguments

# Define Training Parameters
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Step 5: Train the Model

# Define a Compute Metrics Function
metric = load_metric("accuracy")

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return metric.compute(predictions=preds, references=p.label_ids)

# Initialize Trainer and Start Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Step 6: Evaluate the Model

trainer.evaluate()

# Step 7: Save the Model

# Save the Fine-tuned Model
model.save_pretrained("./fine-tuned-bert")
tokenizer.save_pretrained("./fine-tuned-bert")

# Step 8: Inference

# Load and Use the Model for Inference
model = BertForSequenceClassification.from_pretrained("./fine-tuned-bert")
tokenizer = BertTokenizer.from_pretrained("./fine-tuned-bert")

inputs = tokenizer("This is a test sentence.", return_tensors="pt")
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

# Step 9: Fine-Tuning Tips (These are suggestions rather than code)
# - Experiment with different learning rates, typically starting from 2e-5 to 5e-5.
# - Use a batch size that fits within your GPU memory. If memory is an issue, try gradient accumulation.
# - Generally, 2-4 epochs are sufficient for fine-tuning.
# - Regularly evaluate your model during training to monitor overfitting and adjust hyperparameters accordingly.



