## 📧 Email Spam Classifier
A powerful email spam classification system powered by deep learning. This application uses a neural network trained on your dataset to accurately distinguish between spam and legitimate (ham) emails.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ems-classifier.streamlit.app/)

## ✨ Features
Instant Classification: Paste any email and get immediate results

High Accuracy: Neural network achieves ~98% accuracy on test data

Confidence Scores: See probability percentages for both spam and ham classifications

Clean Interface: Simple, professional UI with clear visual feedback

Fast Inference: Results in under 1 second

GPU Support: Automatically utilizes NVIDIA GPU if available

## 🧠 Model Architecture
The system uses a deep neural network with:

Text Preprocessing: TF-IDF vectorization with 3000 features

Input Layer: 3000 features (TF-IDF)

Hidden Layers: 256 → 128 → 64 neurons

Dropout: 30% regularization to prevent overfitting

Output Layer: 2 neurons (spam/ham) with softmax

## Text Preprocessing Pipeline:
Lowercase conversion

HTML tag removal

URL and email address removal

Special character removal

Stop words removal

## 🎯 Usage Guide
Start the app: streamlit run app.py

Wait for model training: The first run will train on your dataset (takes 30-60 seconds)

Enter an email: Paste any email text in the input area

Click "Classify Email": Get instant results with confidence scores

## Interpret results:

🚫 SPAM (red box): High probability of being spam

✅ HAM (green box): Legitimate email

Confidence percentage shows reliability

## 📧 Contact

## Dr. Mahroona Laraib

Project Link: https://github.com/drmahroona/Email-Spam-Classification
