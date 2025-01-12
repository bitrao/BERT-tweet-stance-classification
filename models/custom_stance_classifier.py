import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AdamW
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.max_val_metric = 0

    def early_stop(self, val_metric):
        if val_metric > self.max_val_metric:
            self.max_val_metric = val_metric
            self.counter = 0
        elif val_metric < self.max_val_metric:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class EnhancedClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(input_dim // 2, num_classes)

    def forward(self, pooled_output):
        x = self.fc1(pooled_output)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class CustomStanceClassifier(nn.Module):
    def __init__(self, model_name, num_classes=3, custom_head=None):
        super(CustomStanceClassifier, self).__init__()
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        if custom_head:
            self.custom_head = custom_head
        self.custom_head = EnhancedClassificationHead(self.backbone.config.hidden_size, num_classes)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # for loss evolution
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token output
        logits = self.custom_head(pooled_output)
        return logits
    
    def train_model(self, train_loader, val_loader, device, num_epochs=3, learning_rate=2e-5, class_weights=[], patience=3, weight_decay=0.01):
        optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        if len(class_weights):
            class_weights=torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.to(device)
        
        best_val_f1 = 0
        
        # apply early stop to prevent overfitting
        early_stopper = EarlyStopper(patience=patience)
        
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                stances = batch['stance'].to(device)
                
                optimizer.zero_grad()
                outputs = self(input_ids, attention_mask)
                loss = criterion(outputs, stances)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            
            self.train_losses.append(avg_train_loss)
            
            # Validation
            self.eval()
            val_preds = []
            val_true = []
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    stances = batch['stance'].to(device)
                    
                    outputs = self(input_ids, attention_mask)
                    loss = criterion(outputs, stances)
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    val_preds.extend(preds)
                    val_true.extend(stances.cpu().numpy())
                    
            avg_val_loss = val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            val_f1 = f1_score(val_true, val_preds, average='macro')
            val_acc = accuracy_score(val_true, val_preds)
            
            print(f'Epoch {epoch + 1}:')
            print(f'Average training loss: {avg_train_loss}')
            print(f'Average val loss: {avg_val_loss}')
            print(f'Validation F1: {val_f1:.4f}')
            print(f'Validation Accuracy: {val_acc:.4f}')
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({'model_state_dict': self.state_dict(), 
                            'model_name': self.model_name},
                           f'models/{self.model_name}.pt')
                print(f'New best model saved with F1: {val_f1:.4f}')
            
            if early_stopper.early_stop(val_f1):             
                break
                
        print(f'Best model has F1: {best_val_f1:.4f}')
        self.plot_losses()
        
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        plt.plot(self.train_losses, label='Training Loss', alpha=0.7)
        
        # Plot evaluation loss if available
        if self.val_losses:
            # Create x-axis points for eval losses (assuming eval_steps=100)
            eval_x = np.linspace(0, len(self.train_losses), len(self.val_losses))
            plt.plot(eval_x, self.val_losses, label='Validation Loss', alpha=0.7)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} Training and Validation Loss Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.show()
        plt.close()
                
    @classmethod
    def load_model(cls, path, device='cuda'):
        """Load a saved model"""
        # Load the saved state
        model_state = torch.load(path, map_location=device)
        
        # Create new model instance with saved configuration
        model = cls(model_name=model_state['model_name'])
        
        # Load the state dict
        model.load_state_dict(model_state['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        return model
    
    def predict(self, text, target, device='cuda', max_length=128, raw=False):
        """
        Make prediction for a single text-target pair
        
        Args:
            text (str): The text to analyze
            target (str): The target topic
            device (str): Device to run prediction on ('cuda' or 'cpu')
            max_length (int): Maximum sequence length
        
        Returns:
            dict: Dictionary containing stance prediction and probabilities
        """
        # Ensure model is in evaluation mode
        self.eval()
        
        # Move model to specified device
        self = self.to(device)
        
        # Combine text and target
        combined_input = f"{text} [SEP] {target}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_input,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
            
        return {
            'stance': self.stance_map[prediction],
            'probabilities': {
                'FAVOR': probabilities[0][0].item(),
                'AGAINST': probabilities[0][1].item(),
                'NONE': probabilities[0][2].item()
            }
        }
    
    def predict_batch(self, texts, targets, device='cuda', max_length=128, batch_size=16):
        """
        Make predictions for multiple text-target pairs
        
        Args:
            texts (list): List of texts to analyze
            targets (list): List of corresponding targets
            device (str): Device to run prediction on
            max_length (int): Maximum sequence length
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of prediction dictionaries
        """
        self.eval()
        self = self.to(device)
        
        predictions = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_targets = targets[i:i + batch_size]
            
            # Combine text and target for each pair
            combined_inputs = [f"{text} [SEP] {target}" 
                             for text, target in zip(batch_texts, batch_targets)]
            
            # Tokenize
            encodings = self.tokenizer(
                combined_inputs,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_stances = torch.argmax(outputs, dim=1)
            
            # Convert predictions to dictionaries
            for j in range(len(batch_texts)):
                predictions.append({
                    'stance': self.stance_map[predicted_stances[j].item()],
                    'probabilities': {
                        'FAVOR': probabilities[j][0].item(),
                        'AGAINST': probabilities[j][1].item(),
                        'NONE': probabilities[j][2].item()
                    }
                })
        
        return predictions