import pandas as pd
import torch
from transformers import DistilBertTokenizer
from main import load_model, preprocess_text, TweetDataset, set_seed
from torch.utils.data import DataLoader

def predict_with_saved_model(model_dir, new_tweets_path, output_path):
    """
    Load a saved model and use it to predict on new data.
    
    Args:
        model_dir (str): Directory where the model is saved
        new_tweets_path (str): Path to the new tweets CSV file
        output_path (str): Path to save the predictions
    
    Returns:
        pd.DataFrame: DataFrame with tweet IDs and predictions
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the saved model and tokenizer
    model, tokenizer = load_model(model_dir, device)
    print("Model loaded successfully!")
    
    # Load new tweets
    new_tweets = pd.read_csv(new_tweets_path, encoding='latin-1')
    print(f"Loaded {len(new_tweets)} new tweets for prediction")
    
    # Preprocess tweets
    new_tweets['processed_tweet'] = new_tweets['Tweet'].apply(preprocess_text)
    
    # Create dataset and dataloader
    test_dataset = TweetDataset(
        tweets=new_tweets['processed_tweet'].values,
        labels=None,
        tokenizer=tokenizer
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16
    )
    
    # Make predictions
    model.eval()
    predictions = []
    
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
    
    # Create submission file
    submission = pd.DataFrame({
        'TweetID': new_tweets['TweetID'],
        'Label': predictions
    })
    
    # Save predictions
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    return submission

def predict_single_tweet(model_dir, tweet_text):
    """
    Use the saved model to predict the class of a single tweet.
    
    Args:
        model_dir (str): Directory where the model is saved
        tweet_text (str): The text of the tweet to classify
    
    Returns:
        int: Predicted class (0 or 1)
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the saved model and tokenizer
    model, tokenizer = load_model(model_dir, device)
    
    # Preprocess the tweet
    processed_tweet = preprocess_text(tweet_text)
    
    # Tokenize the tweet
    encoding = tokenizer.encode_plus(
        processed_tweet,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    # Get prediction probability
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    confidence = probabilities[0][prediction].item()
    
    print(f"Tweet: {tweet_text}")
    print(f"Processed: {processed_tweet}")
    print(f"Prediction: Class {prediction} (confidence: {confidence:.4f})")
    print(f"Class 0: Not doing physical activity, Class 1: Doing physical activity")
    
    return prediction, confidence

if __name__ == "__main__":
    # Example usage for batch prediction
    # predict_with_saved_model(
    #     model_dir='tweet_classifier_model',
    #     new_tweets_path='WN25_data/WN25_PA_test_tweets.txt',
    #     output_path='new_predictions.csv'
    # )
    
    # Example usage for single tweet prediction
    example_tweets = [
        "Just finished a 5K run and feeling great!",
        "Watching the basketball game on TV tonight",
        "Did my morning yoga and meditation routine",
        "Looking forward to the football match this weekend",
        "I'm playing video games all day long! So exhausted but proud"
    ]
    
    print("Testing the model with example tweets:\n")
    for tweet in example_tweets:
        predict_single_tweet('tweet_classifier_model', tweet)
        print("-" * 50) 

    # Set seed for reproducibility
    set_seed(42)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
