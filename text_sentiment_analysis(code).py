import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SimpleRNN
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

#Load dataset
data = pd.read_csv(r"C:\Users\munee\Downloads\data.csv")

# Check the column names
print("Dataset Columns:", data.columns)

# Preprocess text data
texts = data["Sentence"].values
labels = data["Sentiment"].values

# Encode sentiment labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels) 

# Tokenize text
tokenizer = Tokenizer(num_words=10000)  # Limit vocabulary to 10,000 words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to ensure uniform input size
max_length = 100  # Maximum sequence length
X = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

# Convert labels to categorical (one-hot encoding)
y = tf.keras.utils.to_categorical(labels, num_classes=3)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),  # Embedding layer
    LSTM(128, return_sequences=False),  # LSTM layer
    Dropout(0.5),  # Dropout for regularization
    Dense(64, activation="relu"),  # Fully connected layer
    Dense(3, activation="softmax")  # Output layer for 3 classes
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

#Predict sentiment for new text
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post", truncating="post")
    prediction = model.predict(padded_sequence)
    sentiment = np.argmax(prediction)
    return "Positive" if sentiment == 2 else "Neutral" if sentiment == 1 else "Negative"

text = "The prices for the company has increased"
print(f"Sentiment: {predict_sentiment(text)}")