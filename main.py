import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
import logging
import os
import random
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seed for reproducibility
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =  False
    os.environ['PYTHONHASHSEED'] = str(seed_value)

# Text preprocessing
def preprocess_text(text):
    """Clean and preprocess tweet text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Custom Dataset class
class TweetDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_len=128):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, idx):
        tweet = str(self.tweets[idx])
        label = self.labels[idx] if self.labels is not None else -1
        
        encoding = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'tweet_text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load and prepare data
def load_data(training_tweets_path, training_labels_path, test_tweets_path=None):
    """Load and prepare the data for training and testing."""
    # Read training data
    training_tweets = pd.read_csv(training_tweets_path, encoding='latin-1')
    training_labels = pd.read_csv(training_labels_path, encoding='latin-1')
    
    # Merge training tweets with their labels
    train_data = pd.merge(training_tweets, training_labels, on='TweetID')
    
    # Preprocess tweets
    train_data['processed_tweet'] = train_data['Tweet'].apply(preprocess_text)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(
        train_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=train_data['Label']
    )
    
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    
    # Load test data if provided
    if test_tweets_path:
        test_tweets = pd.read_csv(test_tweets_path, encoding='latin-1')
        test_tweets['processed_tweet'] = test_tweets['Tweet'].apply(preprocess_text)
        return train_df, val_df, test_tweets
    
    return train_df, val_df

# Training function
def train_model(model, train_dataloader, val_dataloader, device, epochs=4):
    """Train the DistilBERT model."""
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0
    best_model = None
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_f1 = f1_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds)
        val_recall = recall_score(val_true, val_preds)
        
        logger.info(f"Validation F1: {val_f1:.4f}")
        logger.info(f"Validation Precision: {val_precision:.4f}")
        logger.info(f"Validation Recall: {val_recall:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict().copy()
            logger.info(f"New best model with F1: {best_val_f1:.4f}")
    
    # Load best model
    if best_model:
        model.load_state_dict(best_model)
    
    return model, best_val_f1

# Prediction function
def predict(model, test_dataloader, device):
    """Generate predictions using the trained model."""
    model.eval()
    predictions = []
    tweet_ids = []
    
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
    
    return predictions

# Main function to run the entire pipeline
def run_classifier(training_tweets_path, training_labels_path, test_tweets_path, output_path, batch_size=16, epochs=4):
    """Run the entire classification pipeline."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    train_df, val_df, test_df = load_data(training_tweets_path, training_labels_path, test_tweets_path)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = TweetDataset(
        tweets=train_df['processed_tweet'].values,
        labels=train_df['Label'].values,
        tokenizer=tokenizer
    )
    
    val_dataset = TweetDataset(
        tweets=val_df['processed_tweet'].values,
        labels=val_df['Label'].values,
        tokenizer=tokenizer
    )
    
    test_dataset = TweetDataset(
        tweets=test_df['processed_tweet'].values,
        labels=None,
        tokenizer=tokenizer
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    
    model.to(device)
    
    # Train model
    model, best_val_f1 = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        epochs=epochs
    )
    
    # Generate predictions
    predictions = predict(model, test_dataloader, device)
    
    # Create submission file
    submission = pd.DataFrame({
        'TweetID': test_df['TweetID'],
        'Label': predictions
    })
    
    submission.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return model, best_val_f1, submission

# Class distribution analysis
def analyze_class_distribution(labels):
    """Analyze the distribution of classes in the dataset."""
    class_counts = labels.value_counts()
    total = len(labels)
    
    logger.info("Class distribution:")
    for label, count in class_counts.items():
        percentage = (count / total) * 100
        logger.info(f"Class {label}: {count} samples ({percentage:.2f}%)")
    
    return class_counts

# Function to handle class imbalance
def handle_class_imbalance(train_df):
    """Handle class imbalance using class weights."""
    class_counts = train_df['Label'].value_counts()
    total = len(train_df)
    
    # Calculate class weights
    class_weights = {
        label: total / (len(class_counts) * count)
        for label, count in class_counts.items()
    }
    
    logger.info(f"Class weights: {class_weights}")
    
    return torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float)

# Save and load model
def save_model(model, tokenizer, output_dir):
    """Save the trained model and tokenizer."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

def load_model(model_dir, device):
    """Load a trained model and tokenizer."""
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    
    model.to(device)
    logger.info(f"Model loaded from {model_dir}")
    
    return model, tokenizer