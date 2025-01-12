from data.stance_dataset import StanceDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import re
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy


# Compile regex for emoji removal once
emoji_pattern = re.compile(
    r"[\U0001F600-\U0001F64F]|"  # emoticons
    r"[\U0001F300-\U0001F5FF]|"  # symbols & pictographs
    r"[\U0001F680-\U0001F6FF]|"  # transport & map symbols
    r"[\U0001F1E0-\U0001F1FF]|"  # flags (iOS)
    r"[\U00002702-\U000027B0]|"  # Dingbats
    r"[\U000024C2-\U0001F251]",  # Enclosed characters
    flags=re.UNICODE,
)

pun =  """!"$%&'()*+,-./:;<=>?[\\]^`{|}~"""

stop_words = set(stopwords.words('english'))

def deEmojify(text):
    return emoji_pattern.sub(r"", text)

def preprocess_tweet(tweet):
    # Initialize tags set to avoid duplicates
    tags = set()
    
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    
    # Remove non-ASCII characters
    tweet = tweet.encode("ascii", "ignore").decode()
    
    # Extract hashtags
    for word in tweet.split():
        if word.startswith('#'):
            tags.add(word.strip(','))
    
    # Remove hashtags and user mentions
    tweet = re.sub(r'\@\w+|\#\w+', '', tweet)
    
    # Replace punctuations with whitespace
    tweet = tweet.translate(str.maketrans(pun, ' ' * len(pun)))
    
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    
    # Remove reserved words
    tweet = re.sub(r"\b(rt|fav)\b", '', tweet)
    
    # Remove words of less than 3 characters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    
    # Remove emojis
    tweet = deEmojify(tweet)
    
    # Remove extra spaces
    tweet = re.sub(r"\s+", " ", tweet).strip()
    
    # Tokenization and stopword removal
    sp = spacy.load('en_core_web_sm')
    sentence = sp(tweet)
    tweet_tokens = [token.lemma_ if token.lemma_ != '-PRON-' else token.lower_ for token in sentence]
    filtered = [w for w in tweet_tokens if w not in stop_words]
    
    # Combine filtered words with hashtags
    filtered.extend(tags)
    
    return " ".join(filtered)



def prepare_data(df, tokenizer, batch_size=8, weighted=False):
    # Convert stance labels to numeric
    stance_map = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}
    df['Stance_Numeric'] = df['Stance'].map(stance_map)
    
    if weighted:
        class_counts = df['Stance_Numeric'].value_counts()

        classes = np.array(class_counts.index)
        class_weights = compute_class_weight('balanced', classes=classes, y=df['Stance_Numeric'])
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = StanceDataset(
        train_df['Tweet'].values,
        train_df['Target'].values,
        train_df['Stance_Numeric'].values,
        tokenizer
    )
    
    val_dataset = StanceDataset(
        val_df['Tweet'].values,
        val_df['Target'].values,
        val_df['Stance_Numeric'].values,
        tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    if weighted:
        return train_loader, val_loader, class_weights
    else:
        return train_loader, val_loader