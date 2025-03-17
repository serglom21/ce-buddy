# sentry_suggestion/train.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split #Import this.

# Load CodeT5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

# Load data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    inputs = [item['code'] for item in data]
    outputs = [item['sentry_code'] for item in data]
    return inputs, outputs

inputs, outputs = load_data('./data/data.json')

# Split data into training and evaluation sets
train_inputs, eval_inputs, train_outputs, eval_outputs = train_test_split(
    inputs, outputs, test_size=0.2, random_state=42
)

# Tokenize data
train_encodings = tokenizer(train_inputs, truncation=True, padding=True, return_tensors="pt")
train_labels = tokenizer(train_outputs, truncation=True, padding=True, return_tensors="pt").input_ids

eval_encodings = tokenizer(eval_inputs, truncation=True, padding=True, return_tensors="pt")
eval_labels = tokenizer(eval_outputs, truncation=True, padding=True, return_tensors="pt").input_ids

# Create PyTorch Datasets
class SentryDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx].clone().detach() for key in self.encodings}
        item['labels'] = torch.tensor(self.labels[idx]).clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentryDataset(train_encodings, train_labels)
eval_dataset = SentryDataset(eval_encodings, eval_labels) #Create the eval dataset.

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=50,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    gradient_accumulation_steps=4,
    fp16=torch.cuda.is_available()
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, #Add the eval dataset here.
    processing_class=tokenizer
)

# Train the model
trainer.train()

# Save the model (optional, but good practice)
trainer.save_model('./results')