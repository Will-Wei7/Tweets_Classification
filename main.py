import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, log_loss
import re
import nltk
from nltk.corpus import stopwords
import logging
import os
import random
import time
from tqdm import tqdm, trange

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
    # Handle None or NaN values
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Handle specific encoding artifacts
    # Replace common problematic character sequences
    encoding_artifacts = [
        ('õæ_õ_', ''),
        ('õ', 'o'), # Replace with closest ASCII equivalent
        ('æ', 'ae'),
        ('ø', 'o'),
        ('ñ', 'n'),
        ('ü', 'u'),
        ('ö', 'o'),
        ('ä', 'a'),
        ('é', 'e'),
        ('è', 'e'),
        ('ê', 'e'),
        ('ë', 'e'),
        ('à', 'a'),
        ('â', 'a'),
        ('î', 'i'),
        ('ï', 'i'),
        ('ô', 'o'),
        ('û', 'u'),
        ('ç', 'c'),
    ]
    for artifact, replacement in encoding_artifacts:
        text = text.replace(artifact, replacement)
    
    # Handle emoji encoding artifacts
    # Replace common emoji encoding patterns
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    
    # Replace emojis with space to separate words properly
    text = emoji_pattern.sub(' ', text)
    
    # Replace HTML entities
    text = re.sub(r'&amp;', ' and ', text)
    text = re.sub(r'&lt;', ' < ', text)
    text = re.sub(r'&gt;', ' > ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Handle common contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)
    
    # Remove non-ASCII characters (more aggressive approach for encoding issues)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Remove special characters and numbers
    # Keep only letters, spaces, and basic punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Function to analyze text before and after preprocessing
def analyze_preprocessing(texts, n_samples=5):
    """
    Analyze the effect of preprocessing on a sample of texts.
    
    Parameters:
    -----------
    texts : list or pandas Series
        The texts to analyze
    n_samples : int, default=5
        Number of samples to show
    
    Returns:
    --------
    DataFrame
        DataFrame showing original and preprocessed texts
    """
    if isinstance(texts, pd.Series):
        # Try to find examples with encoding issues for better analysis
        if len(texts) > 100:
            # Look for texts with potential encoding issues
            potential_issues = texts[texts.str.contains(r'[^\x00-\x7F]', regex=True, na=False)]
            if len(potential_issues) > 0:
                sample_texts = potential_issues.sample(min(n_samples, len(potential_issues))).tolist()
            else:
                sample_texts = texts.sample(min(n_samples, len(texts))).tolist()
        else:
            sample_texts = texts.sample(min(n_samples, len(texts))).tolist()
    else:
        sample_texts = random.sample(texts, min(n_samples, len(texts)))
    
    results = []
    for text in sample_texts:
        processed = preprocess_text(text)
        results.append({
            'Original': text,
            'Preprocessed': processed
        })
    
    return pd.DataFrame(results)

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
def load_data(training_tweets_path, training_labels_path, test_tweets_path=None, show_preprocessing=False):
    """Load and prepare the data for training and testing."""
    # Read training data
    training_tweets = pd.read_csv(training_tweets_path, encoding='latin-1')
    training_labels = pd.read_csv(training_labels_path, encoding='latin-1')
    
    # Merge training tweets with their labels
    train_data = pd.merge(training_tweets, training_labels, on='TweetID')
    
    # Show preprocessing examples if requested
    if show_preprocessing:
        # Try to find examples with encoding issues
        encoding_issues = train_data[train_data['Tweet'].str.contains(r'[^\x00-\x7F]', regex=True, na=False)]
        if len(encoding_issues) > 0:
            logger.info(f"Found {len(encoding_issues)} tweets with potential encoding issues")
            preprocessing_examples = analyze_preprocessing(encoding_issues['Tweet'], n_samples=min(5, len(encoding_issues)))
        else:
            preprocessing_examples = analyze_preprocessing(train_data['Tweet'])
        
        logger.info("Preprocessing examples:")
        for _, row in preprocessing_examples.iterrows():
            logger.info(f"Original: {row['Original']}")
            logger.info(f"Preprocessed: {row['Preprocessed']}")
            logger.info("-" * 50)
    
    # Preprocess tweets
    train_data['processed_tweet'] = train_data['Tweet'].apply(preprocess_text)
    
    logger.info(f"Total dataset size: {len(train_data)}")
    
    # Load test data if provided
    if test_tweets_path:
        test_tweets = pd.read_csv(test_tweets_path, encoding='latin-1')
        test_tweets['processed_tweet'] = test_tweets['Tweet'].apply(preprocess_text)
        return train_data, test_tweets
    
    return train_data

# Function to freeze/unfreeze layers in the model
def set_layer_freezing(model, num_layers_to_freeze=12):
    """
    Freeze or unfreeze layers in the RoBERTa model.
    
    Parameters:
    -----------
    model : RobertaForSequenceClassification
        The model to modify
    num_layers_to_freeze : int
        Number of transformer layers to freeze (0 means unfreeze all)
    """
    # Freeze/unfreeze embeddings
    if num_layers_to_freeze > 0:
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
    else:
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = True
    
    # Freeze/unfreeze transformer layers
    total_layers = len(model.roberta.encoder.layer)
    
    for i in range(total_layers):
        if i < num_layers_to_freeze:
            # Freeze this layer
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
        else:
            # Unfreeze this layer
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = True
    
    # Always unfreeze the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Log the freezing status
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Frozen {frozen_params:,} parameters ({frozen_params/total_params:.1%})")
    logger.info(f"Trainable {trainable_params:,} parameters ({trainable_params/total_params:.1%})")

# Training function with k-fold validation and gradual unfreezing
def train_model_kfold(model, train_dataloader, val_dataloader, device, epochs=4, gradual_unfreeze=True, 
                     warmup_ratio=0.1, weight_decay=0.01, patience=2):
    """Train the RoBERTa model with gradual unfreezing and warmup strategy.
    
    Parameters:
    -----------
    model : RobertaForSequenceClassification
        The model to train
    train_dataloader : DataLoader
        DataLoader for training data
    val_dataloader : DataLoader
        DataLoader for validation data
    device : torch.device
        Device to train on (cuda or cpu)
    epochs : int, default=4
        Number of training epochs
    gradual_unfreeze : bool, default=True
        Whether to use gradual unfreezing
    warmup_ratio : float, default=0.1
        Ratio of total training steps to use for warmup
    weight_decay : float, default=0.01
        Weight decay for regularization
    patience : int, default=2
        Number of epochs to wait for improvement before early stopping
    """
    # Initial freezing - freeze all transformer layers except the last one
    if gradual_unfreeze:
        # Start with all layers frozen except the last transformer layer and classifier
        # RoBERTa base has 12 layers (0-11)
        set_layer_freezing(model, num_layers_to_freeze=11)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],  # Only optimize unfrozen parameters
        lr=2e-5, 
        eps=1e-8,
        weight_decay=weight_decay  # Add weight decay for regularization
    )
    
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    
    # Calculate warmup steps (e.g., 10% of total steps)
    warmup_steps = int(total_steps * warmup_ratio)
    logger.info(f"Using warmup strategy with {warmup_steps} warmup steps ({warmup_ratio:.0%} of total steps)")
    
    # Learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0
    best_model = None
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    # Calculate unfreezing schedule if using gradual unfreezing
    if gradual_unfreeze:
        # We'll unfreeze one layer after each 1/5 of total epochs
        unfreeze_after_epochs = max(1, epochs // 5)
    
    # Use trange for epoch progress
    for epoch in trange(epochs, desc="Epochs"):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        # Gradual unfreezing
        if gradual_unfreeze and epoch > 0 and epoch % unfreeze_after_epochs == 0:
            # Calculate how many layers to keep frozen
            layers_to_freeze = max(0, 11 - (epoch // unfreeze_after_epochs))
            logger.info(f"Unfreezing more layers. Keeping {layers_to_freeze} layers frozen.")
            set_layer_freezing(model, num_layers_to_freeze=layers_to_freeze)
            
            # Update optimizer to include newly unfrozen parameters
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=2e-5 * (0.9 ** (epoch // unfreeze_after_epochs)),  # Reduce learning rate for later unfreezing
                eps=1e-8,
                weight_decay=weight_decay
            )
            
            # Update scheduler with warmup for remaining epochs
            remaining_steps = len(train_dataloader) * (epochs - epoch)
            remaining_warmup_steps = int(remaining_steps * warmup_ratio)
            logger.info(f"Resetting scheduler with {remaining_warmup_steps} warmup steps for remaining {epochs - epoch} epochs")
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=remaining_warmup_steps,
                num_training_steps=remaining_steps
            )
        
        # Training
        model.train()
        train_loss = 0
        
        # Use tqdm for batch progress
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}", leave=False)
        for step, batch in enumerate(progress_bar):
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
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            
            train_loss += loss.item()
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]
            
            # Update progress bar with current loss and learning rate
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
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
        val_loss = 0
        
        # Add progress bar for validation
        val_progress_bar = tqdm(val_dataloader, desc="Validation", leave=False)
        for batch in val_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            
            # Track validation loss
            loss = outputs.loss
            val_loss += loss.item()
            
            # Update progress bar with current validation loss
            val_progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())
        
        # Calculate average validation loss
        val_avg_loss = val_loss / len(val_dataloader)
        
        # Calculate metrics
        val_log_loss = log_loss(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds)
        val_recall = recall_score(val_true, val_preds)
        
        logger.info(f"Validation Log Loss: {val_log_loss:.4f}")
        logger.info(f"Validation Average Loss: {val_avg_loss:.4f}")
        logger.info(f"Validation F1: {val_f1:.4f}")
        logger.info(f"Validation Precision: {val_precision:.4f}")
        logger.info(f"Validation Recall: {val_recall:.4f}")
        
        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict().copy()
            logger.info(f"New best model with F1: {best_val_f1:.4f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            logger.info(f"No improvement in F1 for {no_improvement_count} epochs")
            
            # Early stopping
            if no_improvement_count >= patience:
                logger.info(f"Early stopping after {epoch+1} epochs due to no improvement in validation F1")
                break
    
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
    
    # Add progress bar for prediction
    progress_bar = tqdm(test_dataloader, desc="Generating predictions")
    for batch in progress_bar:
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

# Main function to run the entire pipeline with k-fold cross-validation
def run_classifier_kfold(training_tweets_path, training_labels_path, test_tweets_path, output_path, 
                         batch_size=16, epochs=4, n_splits=5, gradual_unfreeze=True, warmup_ratio=0.1,
                         weight_decay=0.01, dropout=0.1, patience=2):
    """Run the entire classification pipeline with k-fold cross-validation."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    train_data, test_df = load_data(training_tweets_path, training_labels_path, test_tweets_path)
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Lists to store metrics for each fold
    fold_f1_scores = []
    fold_models = []
    
    # Run k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        logger.info(f"Starting fold {fold+1}/{n_splits}")
        
        # Split data for this fold
        train_fold = train_data.iloc[train_idx]
        val_fold = train_data.iloc[val_idx]
        
        logger.info(f"Training set size: {len(train_fold)}")
        logger.info(f"Validation set size: {len(val_fold)}")
        
        # Create datasets
        train_dataset = TweetDataset(
            tweets=train_fold['processed_tweet'].values,
            labels=train_fold['Label'].values,
            tokenizer=tokenizer
        )
        
        val_dataset = TweetDataset(
            tweets=val_fold['processed_tweet'].values,
            labels=val_fold['Label'].values,
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
        
        # Initialize model for this fold with dropout
        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            hidden_dropout_prob=dropout,  # Add dropout to hidden states
            attention_probs_dropout_prob=dropout,  # Add dropout to attention
        )
        
        model.to(device)
        
        # Train model for this fold
        model, val_f1 = train_model_kfold(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device,
            epochs=epochs,
            gradual_unfreeze=gradual_unfreeze,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            patience=patience
        )
        
        # Store results for this fold
        fold_f1_scores.append(val_f1)
        fold_models.append(model.state_dict().copy())
        
        logger.info(f"Fold {fold+1} completed with F1 score: {val_f1:.4f}")
    
    # Calculate average F1 score across all folds
    avg_f1 = sum(fold_f1_scores) / len(fold_f1_scores)
    logger.info(f"Average F1 score across {n_splits} folds: {avg_f1:.4f}")
    
    # Find the best model based on validation F1 score
    best_fold_idx = fold_f1_scores.index(max(fold_f1_scores))
    logger.info(f"Best model from fold {best_fold_idx+1} with F1 score: {fold_f1_scores[best_fold_idx]:.4f}")
    
    # Load the best model for final prediction
    final_model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )
    
    final_model.load_state_dict(fold_models[best_fold_idx])
    final_model.to(device)
    
    # Create test dataset
    test_dataset = TweetDataset(
        tweets=test_df['processed_tweet'].values,
        labels=None,
        tokenizer=tokenizer
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )
    
    # Generate predictions
    predictions = predict(final_model, test_dataloader, device)
    
    # Create submission file
    submission = pd.DataFrame({
        'TweetID': test_df['TweetID'],
        'Label': predictions
    })
    
    submission.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return final_model, avg_f1, submission

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
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    
    model.to(device)
    logger.info(f"Model loaded from {model_dir}")
    
    return model, tokenizer

# Function to run directly from notebook
def run_classifier(training_tweets_path, training_labels_path, test_tweets_path, output_path, 
                     batch_size=16, epochs=8, n_splits=5, gradual_unfreeze=True, warmup_ratio=0.1,
                     weight_decay=0.01, dropout=0.1, patience=2):
    """
    Run the classifier directly from a notebook without command line arguments.
    
    Parameters:
    -----------
    training_tweets_path : str
        Path to the training tweets CSV file
    training_labels_path : str
        Path to the training labels CSV file
    test_tweets_path : str
        Path to the test tweets CSV file
    output_path : str
        Path to save the predictions CSV file
    batch_size : int, default=16
        Batch size for training
    epochs : int, default=8
        Number of training epochs
    n_splits : int, default=5
        Number of folds for cross-validation
    gradual_unfreeze : bool, default=True
        Whether to use gradual unfreezing during training
    warmup_ratio : float, default=0.1
        Ratio of total training steps to use for warmup
    weight_decay : float, default=0.01
        Weight decay for regularization
    dropout : float, default=0.1
        Dropout rate for regularization
    patience : int, default=2
        Number of epochs to wait for improvement before early stopping
    
    Returns:
    --------
    tuple
        (final_model, avg_f1, submission)
    """
    return run_classifier_kfold(
        training_tweets_path=training_tweets_path,
        training_labels_path=training_labels_path,
        test_tweets_path=test_tweets_path,
        output_path=output_path,
        batch_size=batch_size,
        epochs=epochs,
        n_splits=n_splits,
        gradual_unfreeze=gradual_unfreeze,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        dropout=dropout,
        patience=patience
    )

# If the script is run directly
if __name__ == "__main__":
    import argparse
    import sys
    
    # Check if running in Jupyter notebook
    is_notebook = 'ipykernel' in sys.modules
    
    if is_notebook:
        logger.info("Running in Jupyter notebook environment. Please use run_classifier() function instead of command line arguments.")
        logger.info("Example: model, f1, submission = run_classifier('train_tweets.csv', 'train_labels.csv', 'test_tweets.csv', 'predictions.csv')")
    else:
        parser = argparse.ArgumentParser(description='Run tweet classification with k-fold cross-validation')
        parser.add_argument('--train_tweets', type=str, required=True, help='Path to training tweets CSV')
        parser.add_argument('--train_labels', type=str, required=True, help='Path to training labels CSV')
        parser.add_argument('--test_tweets', type=str, required=True, help='Path to test tweets CSV')
        parser.add_argument('--output', type=str, required=True, help='Path to save predictions CSV')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
        parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation')
        parser.add_argument('--gradual_unfreeze', type=bool, default=True, help='Whether to use gradual unfreezing during training')
        parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Ratio of total training steps to use for warmup')
        parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for regularization')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for regularization')
        parser.add_argument('--patience', type=int, default=2, help='Number of epochs to wait for improvement before early stopping')
        
        args = parser.parse_args()
        
        run_classifier_kfold(
            training_tweets_path=args.train_tweets,
            training_labels_path=args.train_labels,
            test_tweets_path=args.test_tweets,
            output_path=args.output,
            batch_size=args.batch_size,
            epochs=args.epochs,
            n_splits=args.k_folds,
            gradual_unfreeze=args.gradual_unfreeze,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience
        )