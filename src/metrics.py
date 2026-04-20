from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import logging
from src.train import preprare_hf_datasets,train_model
from src.preprocessing import prepare_category_labels,prepare_priority_labels

df,label2id,id2label=prepare_category_labels(df)
df,pri2id,id2pri=prepare_priority_labels(df)

train_dataset,test_dataset=prepare_hf_datasets(train_df,test_df,tokenizer)
trainer=train_model(model,train_dataset,test_dataset)
    
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger=logging.getLogger(__name__)

def evaluate_model(trainer,test_dataset,test_labels_category,test_labels_priority,categories,priorities):
    try:
        predictions=trainer.predict(test_dataset)
        logits=predictions.predictions
        num_categories=len(categories)
        pred_category=np.argmax(logits[:, :num_categories],axis=-1)
        pred_priority=np.argmax(logits[:, num_categories:],axis=-1)
        
        precision_cat,recall_cat,f1_cat, support_cat=precision_recall_fscore_support(test_labels_category,pred_category,average=None, zero_division=0)
        
        precision_pri,recall_pri,f1_pri,support_pri=precision_recall_fscore_support(test_labels_priority, pred_priority, average=None, zero_division=0)
        
        logger.info(f"Category Accuracy: {accuracy_score(test_labels_category, pred_category)*100:.2f}%")
        logger.info(f"Priority Accuracy: {accuracy_score(test_labels_priority, pred_priority)*100:.2f}%")

        logger.info("Class\t\tPrecision\tRecall\tF1-Score\tSupport")
        for i,cat in enumerate(categories):
            logger.info(f"{cat:<15}{precision_cat[i]:.4f}\t\t{recall_cat[i]:.4f}\t{f1_cat[i]:.4f}\t\t{support_cat[i]}")
        logger.info(confusion_matrix(test_labels_category, pred_category))
        
        logger.info("Class\t\tPrecision\tRecall\tF1-Score\tSupport")
        for i, pri in enumerate(priorities):
            logger.info(f"{pri:<15}{precision_pri[i]:.4f}\t\t{recall_pri[i]:.4f}\t{f1_pri[i]:.4f}\t\t{support_pri[i]}"))
        logger.info(confusion_matrix(test_labels_priority,pred_priority))
        
        return accuracy_score(test_labels_category,pred_category),accuracy_score(test_labels_priority,pred_priority)
    except Exception as e:
        logger.error(f"An error occurred:{e}")
        raise
if __name__=="__main__":
    evaluate_model(trainer,test_dataset,test_df["category_label"].values,test_df["priority_label"].values,list(category2id.keys()),list(priority2id.keys()))