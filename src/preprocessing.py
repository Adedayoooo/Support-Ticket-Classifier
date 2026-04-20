import pandas as pd 
from pathlib import Path
from typing import List,Dict,Any,Tuple 
from src.config import TRAIN_DATA_PATH

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)    

def load_and_clean_data(filepath:str=TRAIN_DATA_PATH)->pd.DataFrame:
    try:
        df=pd.read_csv(filepath)
        df['ticket_text']=df['ticket_text'].fillna('').astype(str).str.strip()
        df['category']=df['category'].fillna('General Inquiry').astype(str)
        df['priority']=df['priority'].fillna('medium').astype(str)
        df=df[df['ticket_text']!='']
        return df
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise 

def prepare_category_labels(df:pd.DataFrame)->Tuple[pd.DataFrame,Dict[str,int],Dict[int,str]]:
    try:
        categories=sorted(df['category'].unique())
        label2id={label:idx for idx,label in enumerate(categories)}
        id2label={idx:label for label,idx in label2id.items()}
        df=df.copy()
        df['category_label']=df['category'].map(label2id)
        return df,label2id,id2label 
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise

def prepare_priority_labels(df:pd.DataFrame)->Tuple[pd.DataFrame,Dict[str,int],Dict[int,str]]:
    try:
        priorities=sorted(df['priority'].unique())
        pri2id={priority:idx for idx,priority in enumerate(priorities)}
        id2pri={idx:priority for priority,idx in pri2id.items()}
        df=df.copy()
        df['priority_label']=df['priority'].map(pri2id)
        return df,pri2id,id2pri
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise

if __name__=="__main__":
    df=load_and_clean_data(TRAIN_DATA_PATH)
    df,label2id,id2label=prepare_category_labels(df)
    df,pri2id,id2pri=prepare_priority_labels(df)