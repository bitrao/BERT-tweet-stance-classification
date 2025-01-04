from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from sklearn.metrics import f1_score
import numpy as np
import os
from data.stance_dataset import StanceDataset

class StanceClassifier:
    def __init__(self, model_name='roberta-base', num_labels=3):  # 3 labels: FAVOR, AGAINST, NONE
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    
    def tokenize_data(self, texts, targets):
        return self.tokenizer(
            [f"{text} [SEP] {target}" for text, target in zip(texts, targets)],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'f1': f1_score(labels, predictions, average='weighted')}
    
    def train(self, train_texts, train_targets, train_stances, 
              val_texts=None, val_targets=None, val_stances=None,
              batch_size=16, num_epochs=3, output_dir="./stance_model"):
        
        train_dataset = StanceDataset(train_texts, train_targets, train_stances, self.tokenizer)
        val_dataset = StanceDataset(val_texts, val_targets, val_stances, self.tokenizer) if val_texts else None

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="f1"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics
        )
        
        trainer.train()
        self.save_model(output_dir)
    
    def predict(self, texts, targets):
        inputs = self.tokenize_data(texts, targets)
        outputs = self.model(**inputs)
        return np.argmax(outputs.logits.detach().numpy(), axis=1)
    
    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
    @classmethod
    def load_model(cls, model_dir):
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return instance