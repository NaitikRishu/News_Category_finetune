# News Category Classifier

A fine-tuned DistilBERT model for classifying news articles into four categories: World, Sports, Business, and Sci/Tech.

## Model Description

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the [AG News](https://huggingface.co/datasets/ag_news) dataset. It achieves **90.4% accuracy** on the evaluation set.

- **Model Type:** Text Classification
- **Base Model:** distilbert-base-uncased
- **Language:** English
- **License:** Apache 2.0

## Categories

The model classifies news articles into 4 categories:
- **0:** World
- **1:** Sports
- **2:** Business
- **3:** Sci/Tech

## Training Details

### Training Data

The model was trained on a subset of the AG News dataset:
- Training samples: 5,000
- Evaluation samples: 1,000

### Training Hyperparameters

- Learning rate: 2e-5
- Batch size: 8 (train), 8 (eval)
- Number of epochs: 2
- Weight decay: 0.01
- Max sequence length: 128

### Training Results

- **Evaluation Accuracy:** 90.4%
- **Evaluation Loss:** 0.362

## Usage

### Using Transformers Pipeline

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="naitikrishu/news-category-classifier")

text = "The stock market saw a major rise today due to strong economic reports."
result = classifier(text)

print(result)
# Output: [{'label': 'Business', 'score': 0.98}]
```

### Using Model Directly

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("naitikrishu/news-category-classifier")
model = AutoModelForSequenceClassification.from_pretrained("naitikrishu/news-category-classifier")

# Prepare text
text = "Scientists discover new planet in nearby solar system."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()

# Map to category
categories = ["World", "Sports", "Business", "Sci/Tech"]
print(f"Category: {categories[predicted_class]}")
print(f"Confidence: {predictions[0][predicted_class].item():.2%}")
```

## Example Predictions

| Text | Predicted Category |
|------|-------------------|
| "The stock market saw a major rise today due to strong economic reports." | Business |
| "Scientists discover new planet in nearby solar system." | Sci/Tech |
| "Local team wins championship after intense final match." | Sports |
| "United Nations holds emergency meeting on global crisis." | World |

## Limitations

- The model is trained on a subset of AG News and may not generalize well to all news domains
- Performance may vary on news articles with mixed topics
- Best suited for short to medium-length news headlines and articles
- May struggle with very recent events or topics not well represented in the training data

## Intended Use

This model is intended for:
- Automatic categorization of news articles
- Content organization for news aggregation platforms
- Research and educational purposes in NLP and text classification

## How to Cite

```bibtex
@misc{news-category-classifier,
  author = {Your Name},
  title = {News Category Classifier},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/naitikrishu/news-category-classifier}}
}
```

## Model Card Authors

naitikrishu

## Acknowledgments

- Base model: [DistilBERT](https://huggingface.co/distilbert-base-uncased) by Hugging Face
- Dataset: [AG News](https://huggingface.co/datasets/ag_news)
- Framework: [Transformers](https://github.com/huggingface/transformers) by Hugging Face
