"""
Support Ticket Classifier -Inference Script
Author: Adedayo Adebayo
Date: 2026-03-14
"""

import torch
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_inference_tools(model_path: str):
    logger.info(f"Loading resources from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def get_predictions(texts: List[str], tokenizer, model, device) -> List[Dict[str, Any]]:
    id2label = model.config.id2label
    
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)

    results = []
    for text, pred_id, conf in zip(texts, predictions, confidences):
        results.append({
            "text": text,
            "category": id2label[pred_id.item()],
            "confidence": conf.item()
        })
    return results

def main():
    MODEL_PATH = "Adedayo2000/BERTweet-Support-Ticket-Classifier" 
    
    input_ticket = [
        "My account is locked,help!.",
        "The latest update broke the export feature on the dashboard.",
        "The app is crashing when I try to upload a file.",
        "How do I change the notification settings for my team?",
        "I need to reset my password but the reset link isn't working.",
        "The new UI is confusing and hard to navigate.",
        "I was double charged for my subscription this month, please assist."
    ]
    
    tokenizer, model, device = load_inference_tools(MODEL_PATH)
    
    logger.info("Classifying tickets...")
    predictions = get_predictions(input_ticket, tokenizer, model, device)

    print(f"{'TICKET PREVIEW':<60} | {'CATEGORY':<15} | {'CONF'}")
    
    for res in predictions:
        preview = (res['text'][:40] + '...') if len(res['text']) > 40 else res['text']
        print(f"{preview:<60} | {res['category']:<15} | {res['confidence']:.2%}")

if __name__ == "__main__":
    main()