import joblib
import torch
from transformers import AutoTokenizer
from pathlib import Path
from typing import Dict, Any
from src.config import CONFIG

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)    

class TicketInference:
    def __init__(self, model_path: str = "model.joblib"):
        self.model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        self.model = self._load_model()
        
    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        model = joblib.load(self.model_path)
        model.eval()
        return model
    
    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"],
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        num_categories = len(CONFIG.get("num_categories", 5))
        pred_category = torch.argmax(logits[:, :num_categories], dim=-1).item()
        pred_priority = torch.argmax(logits[:, num_categories:], dim=-1).item()
        return {
            "category_id": pred_category,
            "priority_id": pred_priority,
            "confidence_category": round(torch.softmax(logits[0, :num_categories], dim=-1).max().item(), 4),
            "confidence_priority": round(torch.softmax(logits[0, num_categories:], dim=-1).max().item(), 4)
        }
if __name__=="__main__":
    infer = TicketInference()
    test_text = "My internet connection keeps dropping and I was charged twice"
    result = infer.predict(test_text)
    logger.info(f"Input:", test_text)
    logger.info(f"Prediction:", result)