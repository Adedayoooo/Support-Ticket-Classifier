import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,
from datasets import Dataset
from huggingface_hub import HfApi, login
from src.preprocessing import load_and_clean_data,prepare_category_labels,prepare_priority_labels
import joblib
from src.config import CONFIG_PATH,MODEL_PATH,HF_TOKEN_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

token=HF_TOKEN_PATH.read_text().strip() if HF_TOKEN_PATH.exists() else None

CONFIG=json.loads(CONFIG_PATH.read_text())

def split_data(df: pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame]:
    try:
        train_df, test_df = train_test_split(df,test_size=CONFIG["test_size"],random_state=CONFIG["random_state"],stratify=df["category"])
        logger.info(f"Train: {len(train_df)} samples | Test: {len(test_df)} samples")
        return train_df, test_df
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise e

def tokenize_function(examples, tokenizer):
    try:
        return tokenizer(examples["ticket_text"],max_length=CONFIG["max_length"]) padding="max_length",truncation=True) 
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise

def prepare_hf_datasets(train_df, test_df, tokenizer):
    try:
        train_dataset=Dataset.from_pandas(train_df[["ticket_text", "category_label","priority_label"]])
        test_dataset=Dataset.from_pandas(test_df[["ticket_text", "category_label","priority_label"]])
         train_dataset=train_dataset.map(lambda x:tokenize_function(x,tokenizer),batched=True, remove_columns=["ticket_text"])
        test_dataset=test_dataset.map(lambda x: tokenize_function(x,tokenizer), batched=True, remove_columns=["ticket_text"])
        return train_dataset,test_dataset
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise
    
def load_model_and_tokenizer(num_labels_category, num_labels_priority):
    try
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        model=AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"],num_labels=num_labels_category+num_labels_priority,problem_type="multi_label_classification")
        return model,tokenizer
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise

def train_model(model,train_dataset,eval_dataset):
    try:
        training_args=TrainingArguments(output_dir="./results",num_train_epochs=CONFIG["epochs"],per_device_train_batch_size=CONFIG["batch_size"],per_device_eval_batch_size=CONFIG["batch_size"],warmup_steps=50,weight_decay=0.01,logging_steps=50,eval_strategy="epoch",save_strategy="epoch",load_best_model_at_end=True,report_to="none")
        trainer=Trainer(model=model,args=training_args,train_dataset=train_dataset,eval_dataset=eval_dataset)
        logger.info("Training...")
        trainer.train()
        logger.info("Training complete")
        return trainer
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise

def push_model_to_huggingface(model,repo_name:str,token:str=None):
    try:
        logger.info("Logging into Hugging Face Hub...")
        if token:
            login(token=token)
        else:
            login()
        model_filename=MODEL_PATH
        joblib.dump(model,model_filename)
        logger.info(f"Model saved locally as {model_filename}")
        api=HfApi()
        api.upload_file(path_or_fileob=model_filename,path_in_repo=model_filename,repo_if=repo_name,repo_type="model",commit_message="Upload trained support ticket classifier model")
        logger.info(f"Model successfully pushed to Hugging Face Hub: https://huggingface.co/{repo_name}")
    except Exception as e:
        logger.error(f"Failed to push model to Hugging Face: {e}")
        raise

def main():
    try:
        df=load_and_clean_data()
        
        df,label2id,id2label=prepare_category_labels(df)
        
        df,pri2id,id2pri=prepare_priority_labels(df)
        
        train_df,test_df=split_data(df)
        
        train_dataset,test_dataset=prepare_hf_datasets(train_df,test_df,tokenizer)
        
        model,tokenizer=load_model_and_tokenizer(len(label2id),len(pri2id))
        
        trainer=train_model(model,train_dataset,test_dataset)

if __name__ == "__main__":
    main()