# Tweet Physical Activity Classifier

This project implements a text classifier to classify tweets into two classes:
- **Class '1'**: If the tweet mentions the tweeter doing a moderate to high physical activity or how they feel about the mentioned activity
- **Class '0'**: If the tweet does not refer to the tweeter doing a moderate to high physical activity

## Project Structure

- `main.py`: Contains the core functions for data preprocessing, model training, and evaluation
- `main.ipynb`: Jupyter notebook that implements the workflow and visualizes results
- `requirements.txt`: List of required Python packages
- `WN25_data/`: Directory containing the dataset files
  - `WN25_PA_training_tweets.txt`: Training tweets
  - `WN25_PA_training_labels.txt`: Training labels
  - `WN25_PA_test_tweets.txt`: Test tweets
  - `WN25_sample_output.txt`: Sample output format

## Approach

This project uses DistilBERT, a lightweight version of BERT, for transfer learning and fine-tuning to classify tweets. The approach includes:

1. **Data Preprocessing**: Cleaning tweets by removing URLs, user mentions, special characters, etc.
2. **Class Imbalance Handling**: Using class weights to address imbalance in the dataset
3. **Model Training**: Fine-tuning DistilBERT on the tweet classification task
4. **Evaluation**: Using F1 score as the primary evaluation metric
5. **Prediction**: Generating predictions for the test set

## Setup and Usage

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Jupyter notebook:
   ```
   jupyter notebook main.ipynb
   ```

3. Alternatively, you can use the functions in `main.py` directly:
   ```python
   from main import run_classifier
   
   model, best_val_f1, submission = run_classifier(
       training_tweets_path='WN25_data/WN25_PA_training_tweets.txt',
       training_labels_path='WN25_data/WN25_PA_training_labels.txt',
       test_tweets_path='WN25_data/WN25_PA_test_tweets.txt',
       output_path='predictions.csv',
       batch_size=16,
       epochs=4
   )
   ```

## Model Details

- **Architecture**: DistilBERT (distilbert-base-uncased)
- **Optimizer**: AdamW with learning rate 2e-5
- **Learning Rate Schedule**: Linear schedule with warmup
- **Batch Size**: 16
- **Epochs**: 4
- **Max Sequence Length**: 128 tokens

## Performance

The model's performance is evaluated using the F1 score, which is the harmonic mean of precision and recall. This metric is particularly suitable for imbalanced datasets.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- Transformers 4.0+
- scikit-learn 0.24+
- pandas, numpy, matplotlib, seaborn
- NLTK

## License

This project is for educational purposes only.

## Reference
[distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)