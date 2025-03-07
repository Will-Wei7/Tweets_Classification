import pandas as pd

# File paths
training_tweets_path = 'WN25_data/WN25_PA_training_tweets.txt'
training_labels_path = 'WN25_data/WN25_PA_training_labels.txt'
test_tweets_path = 'WN25_data/WN25_PA_test_tweets.txt'

# Read the files with latin-1 encoding
training_tweets = pd.read_csv(training_tweets_path, encoding='latin-1')
training_labels = pd.read_csv(training_labels_path, encoding='latin-1')
test_tweets = pd.read_csv(test_tweets_path, encoding='latin-1')

# Display the first few rows to check the format
print("Training tweets shape:", training_tweets.shape)
print(training_tweets.head())

print("\nTraining labels shape:", training_labels.shape)
print(training_labels.head())

print("\nTest tweets shape:", test_tweets.shape)
print(test_tweets.head()) 