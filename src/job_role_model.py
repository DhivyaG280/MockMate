import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load and preprocess
df = pd.read_csv(r"D:\project work I\dataset\UpdatedResumeDataSet.csv")
df = df[['Resume', 'Category']].dropna()
df = df.rename(columns={'Resume': 'text', 'Category': 'label'})

# Label encoding
label2id = {label: i for i, label in enumerate(df['label'].unique())}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df['label'].map(label2id)

# Train-test split
train_texts, val_texts = train_test_split(df, test_size=0.2, stratify=df['label'])

train_dataset = Dataset.from_pandas(train_texts)
val_dataset = Dataset.from_pandas(val_texts)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label2id))

def tokenize(example):
    return tokenizer(example['text'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir="./job_role_model",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

model.save_pretrained("job_role_model")
tokenizer.save_pretrained("job_role_model")
torch.save(id2label, "job_role_model/id2label.pt")
