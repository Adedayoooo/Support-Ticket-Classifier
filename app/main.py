from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from src.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Support Ticket Classifier API",description="Multi-label classification for category and priority",version="1.0.0")

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

try:
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"],num_labels=CONFIG["num_labels"])
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    tokenizer = None
    model = None

id2category = CONFIG["id2category"]
id2priority = CONFIG["id2priority"]
num_categories = len(id2category)

class TicketInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {
        "message": "Support Ticket Multi-Label Classifier API",
        "status": "live",
        "endpoints": {
            "POST /classify": "Classify ticket into category and priority"
        }
    }

@app.post("/classify")
def classify_ticket(input: TicketInput):
    if not input.text or not input.text.strip():
        return {"error": "Ticket text cannot be empty"}

    if model is None or tokenizer is None:
        return {"error": "Model not loaded"}

    try:
        inputs = tokenizer(
            input.text,
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"],
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits

        category_logits = logits[0, :num_categories]
        priority_logits = logits[0, num_categories:]

        pred_category_idx = torch.argmax(category_logits).item()
        pred_priority_idx = torch.argmax(priority_logits).item()

        confidence_category = torch.softmax(category_logits, dim=-1).max().item()
        confidence_priority = torch.softmax(priority_logits, dim=-1).max().item()

        pred_category = id2category[str(pred_category_idx)]
        pred_priority = id2priority[str(pred_priority_idx)]

        return {
            "text": input.text,
            "category": pred_category,
            "priority": pred_priority,
            "confidence_category": round(confidence_category, 4),
            "confidence_priority": round(confidence_priority, 4)
        }

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"error": "Classification failed", "details": str(e)}