from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Initialize BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer_bert = AutoTokenizer.from_pretrained(model_name)
model_bert = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3
)

# Prepare data for BERT
le_bert = LabelEncoder()
y_train_bert = le_bert.fit_transform(train_s["label"])
y_val_bert = le_bert.transform(val_s["label"])
y_test_bert = le_bert.transform(test_s["label"])

# Create datasets
train_dataset = SentimentDataset(
    train_s["clean_text"].tolist(), y_train_bert, tokenizer_bert
)
val_dataset = SentimentDataset(val_s["clean_text"].tolist(), y_val_bert, tokenizer_bert)
test_dataset = SentimentDataset(
    test_s["clean_text"].tolist(), y_test_bert, tokenizer_bert
)


# Define compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


# Training arguments
training_args = TrainingArguments(
    output_dir="../models/bert_sentiment",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../models/bert_logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Initialize trainer
trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
print("Training BERT model...")
trainer.train()

# Save the model
model_bert.save_pretrained("../models/saved_bert")
tokenizer_bert.save_pretrained("../models/saved_bert")
print("BERT model training completed and saved!")


# Evaluate BERT model
bert_results = trainer.evaluate(test_dataset)
print(f"BERT Test Accuracy: {bert_results['eval_accuracy']:.4f}")
print(f"BERT Test F1-Score: {bert_results['eval_f1']:.4f}")

# Get predictions for detailed classification report
predictions = trainer.predict(test_dataset)
y_pred_bert = np.argmax(predictions.predictions, axis=1)

print("\nBERT Classification Report:")
print(classification_report(y_test_bert, y_pred_bert, target_names=le_bert.classes_))
