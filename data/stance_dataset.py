import torch
from torch.utils.data import Dataset

class StanceDataset(Dataset):
    def __init__(self, texts, targets, stances, tokenizer, max_length=128):
        self.texts = texts
        self.targets = targets
        self.stances = stances
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = str(self.targets[idx])
        
        # Combine text and target with special token
        combined_input = f"{text} [SEP] {target}"
        
        encoding = self.tokenizer(
            combined_input,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'stance': torch.tensor(self.stances[idx], dtype=torch.long)
        }
