## Project Overview
This project is a **Transformer-Based Fake News Detection System** that leverages 
**BERT (Bidirectional Encoder Representations from Transformers)** to classify news 
articles as **Real** or **Fake**. The model is fine-tuned on the ISOT Fake News Dataset 
containing over **44,000 news articles**, enabling it to learn contextual and semantic 
patterns in news content.

The system implements a complete NLP workflow including data preprocessing, tokenization, 
transformer-based feature extraction, model training, evaluation, and inference.

### Key Features
- Fine-tuned **BERT Transformer** for binary text classification.
- Processed and analyzed **44K+ news articles** from the ISOT Fake News Dataset.
- Implemented text preprocessing and **WordPiece tokenization** using Hugging Face Transformers.
- Built using **TensorFlow/Keras** and the Hugging Face ecosystem.
- Achieved **94% test accuracy** on the ISOT dataset.
- Supports inference on custom news articles for real-time fake news detection.

### Technologies Used
- Python
- TensorFlow / Keras
- Hugging Face Transformers
- BERT (bert-base-uncased)
- Scikit-learn
- Pandas
- NumPy

### Model Performance
> Evaluated on the ISOT Fake News Dataset test split. Note: ISOT's real and fake articles 
> originate from distinct sources (Reuters vs. flagged sites), which contributes to 
> higher within-dataset scores. Cross-dataset generalization may vary.

| Metric    | Score |
|-----------|-------|
| Accuracy  | 94%   |
| Precision | 94%   |
| Recall    | 94%   |
| F1-Score  | 94%   |

### Workflow
1. Load and combine real and fake news datasets.
2. Perform text preprocessing and feature preparation.
3. Tokenize text using BERT's WordPiece tokenizer.
4. Fine-tune a pre-trained BERT model for binary classification.
5. Evaluate performance using standard classification metrics.
6. Save the trained model for future inference and deployment.
