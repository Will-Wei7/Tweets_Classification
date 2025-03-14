#!/usr/bin/env python3
"""
Simple script to test a single sentence with the tweet classification model.
"""

import torch
from main import load_model, preprocess_text
import os

def test_sentence(sentence, model_dir="saved_roberta_model"):
    """
    Test a single sentence with the tweet classification model.
    
    Args:
        sentence (str): The sentence to classify
        model_dir (str): Directory containing the model
    
    Returns:
        dict: Classification result
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    try:
        print(f"Loading model from {model_dir}...")
        model, tokenizer = load_model(model_dir, device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Preprocess the text
    processed_text = preprocess_text(sentence)
    print(f"Original text: {sentence}")
    print(f"Processed text: {processed_text}")
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        processed_text,
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
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0][prediction].item()
    
    # Create class labels
    class_labels = {
        0: "Not mentioned doing physical activity",
        1: "Mentioned doing physical activity"
    }
    
    # Display the result
    print("\n" + "-"*60)
    print(f"PREDICTION: {class_labels[prediction]} (Class {prediction})")
    print(f"CONFIDENCE: {confidence:.4f}")
    print(f"CLASS PROBABILITIES:")
    print(f"  - Not Mentioned Physical Activity: {probabilities[0][0].item():.4f}")
    print(f"  - Mentioned Physical Activity: {probabilities[0][1].item():.4f}")
    print("-"*60)
    
    return {
        "original_text": sentence,
        "processed_text": processed_text,
        "prediction": prediction,
        "prediction_label": class_labels[prediction],
        "confidence": confidence,
        "probabilities": {
            "class_0": probabilities[0][0].item(),
            "class_1": probabilities[0][1].item()
        }
    }

if __name__ == "__main__":
    import sys
    
    # Use command line argument if provided, otherwise use the default sentence
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
    else:
        sentence = "I want to play video game tonight"
    
    test_sentence(sentence) 