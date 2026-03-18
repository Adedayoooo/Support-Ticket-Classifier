# Support Ticket Classifier

Multi-class text classification system for customer support tickets using BERTweet.

## Project Overview

This project classifies customer support tickets into 5 categories:
- **Account Access** - Login issues, password resets
- **Billing/Payment** - Charges, refunds, payment issues  
- **Feature Request** - New features, improvements
- **General Inquiry** - Questions, information requests
- **Technical Support** - Bugs, crashes, technical issues

## Performance

- **Model**: BERTweet (vinai/bertweet-base)
- **Accuracy**: 85%+
- **Dataset**: 388 realistic support tickets
- **Training**: 4 epochs on NVIDIA T4 GPU

## Live Demo

Try the model on Hugging Face:
https://huggingface.co/Adedayo2000/BERTweet-Support-Ticket-Classifier

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained('Adedayo2000/BERTweet-Support-Ticket-Classifier')
tokenizer = AutoTokenizer.from_pretrained('Adedayo2000/BERTweet-Support-Ticket-Classifier')

# Classify ticket
text = "I can't login to my account, password reset not working"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
pred_idx = torch.argmax(probs).item()
category = model.config.id2label[pred_idx]
confidence = probs[0][pred_idx].item()

print(f"Category: {category}")
print(f"Confidence: {confidence*100:.1f}%")
```

## Technical Details

### Architecture
- **Base Model**: BERTweet (Twitter-pretrained BERT)
- **Task**: Multi-class sequence classification
- **Max Length**: 128 tokens
- **Batch Size**: 8

### Training Configuration
- **Optimizer**: AdamW (default from huggingface Trainer)
- **Learning Rate**: 5e-5(default from huggingface Trainer)
- **Epochs**: 4
- **Train/Test Split**: 80/20
- **Hardware**: Kaggle (NVIDIA T4 GPU)

### Dataset
- 388 realistic support tickets
- Includes slang, emojis, typos
- 5 categories

## Project Structure

```
support-ticket-classifier/
├── README.md
├── train.py                    # Training pipeline
├── requirements.txt            # Dependencies
└── notebooks/
    └── training.ipynb          # Kaggle training notebook
```

## Installation

```bash
pip install transformers torch pandas scikit-learn huggingface-hub
```

## Key Learnings

This project taught me:
- Production-grade ML pipeline development
- Handling device placement errors in PyTorch
- Label mapping best practices
- Model deployment to Hugging Face Hub
- Senior-level code structure and documentation

## Future Improvements

- [ ] Add FastAPI endpoint for real-time classification
- [ ] Implement confidence threshold for uncertain predictions
- [ ] Add support for multi-label classification
- [ ] Deploy as web application
- [ ] Collect production feedback for model improvements

## 👤 Author

**Adedayo Adebayo** - Aspiring ML Engineer

- 🔗 Hugging Face: https://huggingface.co/Adedayo2000
- 💼 LinkedIn: https://www.linkedin.com/in/adedayo-adebayo-64b23226b

## License

This project is open source and available under the MIT License.

## Acknowledgments

- BERTweet team for the pre-trained model
- Kaggle for free GPU resources
- Hugging Face for model hosting

---

⭐ **Star this repo if you find it helpful!**
