# Chatbot-LSTM

# 🤖 Emotion Classification Chatbot
Welcome to the Emotion Classification Chatbot project! This chatbot is designed to recognize and classify emotions from text input provided by users. Whether you're feeling joy, anger, sadness, or any other emotion, this chatbot can identify and respond accordingly, making the interaction more empathetic and human-like.

The underlying model is built using state-of-the-art deep learning techniques, specifically Long Short-Term Memory (LSTM), to effectively understand and analyze text data for emotion classification.

# 🔍 Project Overview

This chatbot works as follows:

User Input: The user types in a sentence or a statement.

Emotion Classification: The model processes the input and classifies it into one of six emotions:

Joy

Anger

Love

Sadness

Fear

Surprise

Response Generation: Based on the identified emotion, the chatbot responds with a message reflecting the user's emotional state.

# 🧠 How It Works

Data Preprocessing:

The dataset consists of labeled text data containing user inputs and corresponding emotions.

Text Tokenization: The text is split into tokens using a tokenizer, and a vocabulary is built.

Padding & Sequencing: The sentences are padded to ensure uniform input size for the model.

Model Architecture:

LSTM: The model uses an LSTM layer to process the sequence of tokens and capture the context of the text.

Embedding Layer: Converts each word into a dense vector representation for better learning.

Fully Connected Layer: After processing the sequences, the output is passed through a fully connected layer to classify the dominant emotion.

Training & Evaluation:

The model is trained using a Cross-Entropy Loss function, optimizing through Adam optimizer.

The performance is evaluated on a validation set, and predictions are made on the test set to assess its accuracy.


# Result:

<img width="501" alt="Ekran Resmi 2025-02-08 15 20 42" src="https://github.com/user-attachments/assets/86aa7fa0-cc0c-42c1-9689-b930629fb110" />


# 🧠 LSTM Model Architecture:

![image](https://github.com/user-attachments/assets/f09af479-df2a-4856-9ea1-6fd6d73f903b)

