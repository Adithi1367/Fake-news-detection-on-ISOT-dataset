import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification
)

# =====================================
# Load Dataset
# =====================================

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True)

df["content"] = df["title"] + " " + df["text"]

df = df[["content", "label"]]

print("Dataset Shape:", df.shape)

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =====================================
# Load Tokenizer
# =====================================

MODEL_NAME = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# =====================================
# Tokenization
# =====================================

MAX_LEN = 256

train_encodings = tokenizer(
    X_train.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LEN
)

test_encodings = tokenizer(
    X_test.tolist(),
    truncation=True,
    padding=True,
    max_length=MAX_LEN
)

# =====================================
# Create TensorFlow Dataset
# =====================================

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train.values
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test.values
))

BATCH_SIZE = 8

train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# =====================================
# Load BERT Model
# =====================================

model = TFBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# =====================================
# Compile Model
# =====================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=2e-5
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True
)

metric = tf.keras.metrics.SparseCategoricalAccuracy()

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric]
)

# =====================================
# Train
# =====================================

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=3
)

# =====================================
# Evaluate
# =====================================

loss, accuracy = model.evaluate(test_dataset)

print(f"\nAccuracy: {accuracy:.4f}")

# =====================================
# Predictions
# =====================================

predictions = model.predict(test_dataset)

y_pred = np.argmax(
    predictions.logits,
    axis=1
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =====================================
# Save Model
# =====================================

model.save_pretrained("fake_news_bert_tf")
tokenizer.save_pretrained("fake_news_bert_tf")

print("Model Saved Successfully")