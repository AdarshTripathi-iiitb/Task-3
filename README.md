# Sentiment Analysis: Multiclass Classification  

## Project Overview  
This project focuses on developing a sentiment analysis model to classify tweets into one of three sentiment categories:  
- **Negative (-1)**  
- **Neutral (0)**  
- **Positive (+1)**  

Two approaches were explored:  
1. **Support Vector Machine (SVM)** for classical machine learning-based classification.  
2. **Language Model (LLM)** leveraging advanced transformer-based architectures for natural language understanding.  

The final output is a CSV file (`test3.csv`) containing predictions for the provided test dataset.  

---

## Approach 1: Support Vector Machine (SVM)  
### Description  
Support Vector Machine (SVM) was used as a classical machine learning approach for the sentiment classification task. The SVM model was trained using tweet embeddings derived from text features. This approach emphasizes simplicity and efficiency for small to medium-sized datasets.  

### Steps Implemented:  
1. **Data Preprocessing**:  
   - Tweets were cleaned by removing unwanted characters, punctuation, and stopwords.  
   - Tokenization was applied to split text into meaningful words.  

2. **Feature Extraction**:  
   - Used techniques like Term Frequency-Inverse Document Frequency (TF-IDF) or word embeddings to convert text into numerical features.  

3. **Model Training**:  
   - SVM with a linear kernel was trained to classify the sentiments into three categories.  

4. **Evaluation**:  
   - The model was validated using metrics such as accuracy, precision, recall, and F1-score.  

5. **Prediction**:  
   - The trained model was applied to the test dataset, and predictions were stored in `test3.csv`.  

---

## Approach 2: Language Model (LLM)  
### Description  
A transformer-based pre-trained Language Model (LLM) was fine-tuned for the sentiment classification task. This method leverages state-of-the-art architectures to capture contextual meaning in tweets.  

### Steps Implemented:  
1. **Data Preparation**:  
   - Tweets were tokenized using a pre-trained tokenizer (e.g., BERT, GPT, or similar).  
   - Padded and truncated sequences to match the input size required by the model.  

2. **Model Fine-Tuning**:  
   - A pre-trained transformer model (e.g., BERT, RoBERTa) was fine-tuned on the training dataset with a classification head for multiclass output.  
   - The training process used cross-entropy loss and an Adam optimizer with weight decay.  

3. **Evaluation**:  
   - Evaluated the model on the validation set using metrics such as accuracy and F1-score.  

4. **Inference**:  
   - Generated predictions for the test dataset and saved them in `test3.csv`.  

---

## Key Files  
- `SVM_Approach.ipynb`: Jupyter notebook implementing the SVM-based approach.  
- `LLM_Approach.ipynb`: Jupyter notebook implementing the LLM-based approach.  
- `test3.csv`: Output file containing sentiment predictions for the test dataset.  

---

## How to Run  
1. Clone this repository:  
   ```bash  
   git clone <repository-url>  
   cd <repository-directory>  
