import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AdamW

from sklearn.metrics import f1_score, accuracy_score

class StanceClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=3):
        super(StanceClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def train(self, train_loader, val_loader, device, num_epochs=3):
        optimizer = AdamW(model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        model = model.to(device)
        
        best_val_f1 = 0
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                stances = batch['stance'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, stances)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    stances = batch['stance']
                    
                    outputs = model(input_ids, attention_mask)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    val_preds.extend(preds)
                    val_true.extend(stances.numpy())
            
            val_f1 = f1_score(val_true, val_preds, average='macro')
            val_acc = accuracy_score(val_true, val_preds)
            
            print(f'Epoch {epoch + 1}:')
            print(f'Average training loss: {train_loss / len(train_loader)}')
            print(f'Validation F1: {val_f1:.4f}')
            print(f'Validation Accuracy: {val_acc:.4f}')
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), 'best_model.pt')

def main():
    # Initialize tokenizer and model
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = StanceClassifier(model_name)
    
    # Sample usage (replace with your data)
    # df = pd.read_csv('stance_data.csv')
    # train_loader, val_loader = prepare_data(df, tokenizer)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_model(model, train_loader, val_loader, device)