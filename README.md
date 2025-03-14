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

This project uses RoBERTa Base, a robustly optimized BERT pretraining approach, for transfer learning and fine-tuning to classify tweets. The approach includes:

1. **Data Preprocessing**: Cleaning tweets by removing URLs, user mentions, special characters, and handling encoding artifacts
2. **K-Fold Cross-Validation**: Using 5-fold cross-validation to ensure robust model evaluation
3. **Gradual Unfreezing**: Implementing layer-by-layer unfreezing during training to improve fine-tuning
4. **Warmup Strategy**: Using learning rate warmup to stabilize training
5. **Regularization**: Applying dropout and weight decay to prevent overfitting
6. **Early Stopping**: Monitoring validation F1 score to prevent overfitting
7. **Evaluation**: Using F1 score as the primary evaluation metric
8. **Prediction**: Generating predictions for the test set

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
       epochs=8,
       gradual_unfreeze=True,
       warmup_ratio=0.1,
       weight_decay=0.01,
       dropout=0.1
   )
   ```

4. To save the trained model:
   ```python
   from transformers import RobertaTokenizer
   from main import save_model
   
   # Initialize the tokenizer
   tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
   
   # Save the model and tokenizer
   save_model(model, tokenizer, 'saved_roberta_model')
   ```

5. To load a saved model:
   ```python
   import torch
   from main import load_model
   
   # Check for GPU
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   # Load the saved model and tokenizer
   loaded_model, loaded_tokenizer = load_model('saved_roberta_model', device)
   ```

## Model Details

- **Architecture**: RoBERTa Base (roberta-base)
- **Optimizer**: AdamW with learning rate 2e-5
- **Learning Rate Schedule**: Linear schedule with warmup
- **Batch Size**: 16
- **Epochs**: 8
- **Max Sequence Length**: 128 tokens
- **Dropout Rate**: 0.1
- **Weight Decay**: 0.01
- **Gradual Unfreezing**: Starting with 11 frozen layers, gradually unfreezing during training
- **Early Stopping**: Based on validation F1 score with patience of 2 epochs

## Performance

The model's performance is evaluated using the F1 score, which is the harmonic mean of precision and recall. This metric is particularly suitable for imbalanced datasets. K-fold cross-validation provides a more robust estimate of model performance.

## Requirements

- Python 3.6+
- PyTorch 1.7+
- Transformers 4.0+
- scikit-learn 0.24+
- pandas, numpy, matplotlib, seaborn
- NLTK
- tqdm

## License

This project is for educational purposes only.

## Reference

[distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)
[RoBERTa](https://huggingface.co/docs/transformers/en/model_doc/roberta)
[RoBERTa Paper](https://arxiv.org/abs/1907.11692)
[Utilities for Tokenizers](https://huggingface.co/docs/transformers/v4.49.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus)
[Investigating the Characteristics of a Transformer in a Few-Shot Setup: Does Freezing Layers in RoBERTa Help?](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://aclanthology.org/2022.blackboxnlp-1.19.pdf)
[ULMFiT: Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) (for gradual unfreezing technique)
