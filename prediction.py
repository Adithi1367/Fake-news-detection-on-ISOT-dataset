import tensorflow as tf
from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification
)

MODEL_PATH = "fake_news_bert_tf"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

model = TFBertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

def predict_news(news_text):

    encoded = tokenizer(
        news_text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=256
    )

    output = model(encoded)

    prediction = tf.argmax(
        output.logits,
        axis=1
    ).numpy()[0]

    if prediction == 1:
        return "REAL NEWS"
    else:
        return "FAKE NEWS"


sample_news = """
Scientists announce breakthrough in cancer treatment.
"""

print(predict_news(sample_news))