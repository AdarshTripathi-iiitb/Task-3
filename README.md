# Sentiment Classification using LLM and SVM

This project involves sentiment classification of textual data using two approaches:  
1. **Pretrained Language Model (LLM)**: Fine-tuned DistilBERT for multi-class sentiment classification.  
2. **Support Vector Machine (SVM)**: A classical machine learning approach using TF-IDF features for the same task.

## Features

- Uses **DistilBERT**, a lightweight transformer model, for deep learning-based sentiment analysis.
- Implements **SVM** with TF-IDF vectorization for a traditional machine learning approach.
- Compares the accuracy of both models.
- Provides predictions for unseen test data.

---

## Dataset
The dataset includes text samples and their corresponding sentiment categories.  

### Preprocessing:
1. Removed missing values in `Text` and `category` columns.
2. Converted `category` to numeric format.
3. Incremented sentiment labels by `1` for compatibility with training pipelines.

---

## Models

### **1. LLM (DistilBERT-based)**
- Pretrained Model: `distilbert-base-uncased`.
- Tokenized the text data with a maximum length of 128 tokens.
- Fine-tuned the model with:
  - **Optimizer**: AdamW.
  - **Loss Function**: CrossEntropyLoss.
  - **Batch Size**: 16.
  - **Epochs**: 4 with early stopping.
- **Accuracy Achieved**: **87.49%**

### **2. SVM**
- Extracted TF-IDF features with a maximum of 5000 dimensions.
- Trained a linear Support Vector Machine.
- **Accuracy Achieved**: **85%**

---

## Steps to Run the Project

### Prerequisites
1. Python 3.7 or above.
2. Install required libraries:
   ```bash
   pip install pandas scikit-learn transformers torch tqdm
