from data.stance_dataset import StanceDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(df, tokenizer, batch_size=16):
    # Assuming df has columns: 'text', 'target', 'stance'
    # Convert stance labels to numeric
    stance_map = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}
    df['stance_numeric'] = df['stance'].map(stance_map)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = StanceDataset(
        train_df['text'].values,
        train_df['target'].values,
        train_df['stance_numeric'].values,
        tokenizer
    )
    
    val_dataset = StanceDataset(
        val_df['text'].values,
        val_df['target'].values,
        val_df['stance_numeric'].values,
        tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader