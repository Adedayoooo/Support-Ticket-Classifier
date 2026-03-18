"""
Support Ticket Classifier
Author: Adedayo Adebayo
Date: 2026-03-14
Description: Multi-class text classification for customer support tickets using BERTweet
"""

import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from huggingface_hub import HfApi, login

CONFIG = {
    "model_name": "vinai/bertweet-base",
    "test_size": 0.2,
    "random_state": 99,
    "batch_size": 8,
    "epochs": 4,
    "max_length": 128,
}

def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

def load_dataset(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    logger.info(f"Loading dataset from {filepath}")
    ticket_dataframe = pd.read_csv(filepath)
    ticket_dataframe = clean_dataset(ticket_dataframe)
    categories = sorted(ticket_dataframe['category'].unique().tolist())
    logger.info(f"Loaded {len(ticket_dataframe)} samples with {len(categories)} categories")
    return ticket_dataframe, categories

def clean_dataset(ticket_dataframe: pd.DataFrame) -> pd.DataFrame:
    ticket_dataframe['ticket_text'] = ticket_dataframe['ticket_text'].fillna('').astype(str)
    ticket_dataframe['category'] = ticket_dataframe['category'].fillna('General Inquiry')
    ticket_dataframe = ticket_dataframe[ticket_dataframe['ticket_text'].str.strip() != '']
    return ticket_dataframe

def create_labels(ticket_dataframe: pd.DataFrame, categories: List[str]) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(categories)}
    id2label = {idx: label for label, idx in label2id.items()}
    ticket_dataframe['label'] = ticket_dataframe['category'].map(label2id)
    return ticket_dataframe, label2id, id2label

def split_dataset(ticket_dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ticket = ticket_dataframe['ticket_text'].values
    category = ticket_dataframe['label'].values
    ticket_train, ticket_test, category_train, category_test = train_test_split(ticket, category, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=category)
    logger.info(f"Train: {len(ticket_train)}, Test: {len(ticket_test)}")
    return ticket_train, ticket_test, category_train, category_test

class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, texts: np.ndarray, labels: np.ndarray, tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(str(self.texts[idx]), add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }

def create_datasets(ticket_train: np.ndarray, ticket_test: np.ndarray, category_train: np.ndarray, category_test: np.ndarray, tokenizer: AutoTokenizer):
    return TicketDataset(ticket_train, category_train, tokenizer, CONFIG['max_length']), TicketDataset(ticket_test, category_test, tokenizer, CONFIG['max_length'])

def load_model_and_tokenizer(num_labels: int, label2id: Dict[str, int], id2label: Dict[int, str]):
    logger.info(f"Loading model: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG['model_name'], num_labels=num_labels, id2label=id2label, label2id=label2id)
    return model, tokenizer

def train_model(model, train_dataset, test_dataset) -> Trainer:
    logger.info("Training started")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=CONFIG['epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)
    trainer.train()
    logger.info("Training complete")
    return trainer

def evaluate_model(trainer, test_dataset, category_test: np.ndarray, categories: List[str]) -> float:
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    accuracy = accuracy_score(category_test, preds)
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    print("CLASSIFICATION REPORT")
    print(classification_report(category_test, preds, target_names=categories))
    print(confusion_matrix(category_test, preds))
    return accuracy

def push_to_huggingface(model, tokenizer, repo_name: str = "BERTweet-Support-Ticket-Classifier") -> str:
    logger.info("Pushing to HuggingFace Hub...")
    
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        logger.info("Logging in to HuggingFace hub...")
        login(token=hf_token)
        logger.info("Logged in to HuggingFace")
        
        api = HfApi()
        user_info = api.whoami(token=hf_token)
        username = user_info['name']
        full_repo_name = f"{username}/{repo_name}"
        
        logger.info(f"Uploading to: {full_repo_name}")
        
        model.push_to_hub(repo_name, token=hf_token)
        tokenizer.push_to_hub(repo_name, token=hf_token)
        
        model_url = f"https://huggingface.co/{full_repo_name}"
        
        logger.info("Model Uploaded Successfully!")
        logger.info(f"URL: {model_url}")
        
        return model_url
        
    except Exception as e:
        logger.error(f"Failed to push due to the following error: {e}")
        raise

def predict_single(text: str, model, tokenizer, id2label: Dict[int, str]) -> Dict[str, Any]:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=CONFIG['max_length'], padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_idx = torch.argmax(probs).item()
    return {'category': id2label[pred_idx], 'confidence': probs[0][pred_idx].item()}

def run_test_predictions(model, tokenizer, id2label: Dict[int, str]) -> None:
    test_samples = [""]
    print("TEST PREDICTIONS")
    for sample in test_samples:
        result = predict_single(sample, model, tokenizer, id2label)
        print(f"\n{sample} → {result['category']} ({result['confidence']*100:.1f}%)")

def main() -> None:
    logger.info("Loading pipeline...")
    ticket_dataframe, categories = load_dataset('/kaggle/input/datasets/adedayoadebayo23/support-ticket/support_tickets_realistic.csv')
    ticket_dataframe, label2id, id2label = create_labels(ticket_dataframe, categories)
    ticket_train, ticket_test, category_train, category_test = split_dataset(ticket_dataframe)
    model, tokenizer = load_model_and_tokenizer(len(categories), label2id, id2label)
    train_dataset, test_dataset = create_datasets(ticket_train, ticket_test, category_train, category_test, tokenizer)
    trainer = train_model(model, train_dataset, test_dataset)
    accuracy = evaluate_model(trainer, test_dataset, category_test, categories)
    push_to_huggingface(model, tokenizer, repo_name="BERTweet-Support-Ticket-Classifier")
    run_test_predictions(model, tokenizer, id2label)
    logger.info(f"COMPLETED | Accuracy: {accuracy*100:.2f}%")
if __name__ == "__main__":
    main()  