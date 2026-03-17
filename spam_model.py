"""
Email Spam Classification Model
PyTorch neural network for classifying emails as spam or ham
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print("⚠️ Using CPU (GPU not available)")

class SpamClassifierNN(nn.Module):

    def __init__(self, input_dim):
        super(SpamClassifierNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        return self.network(x)

class SpamClassifier:
    """
    Main classifier class handling text preprocessing, training, and inference
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.label_map = {'ham': 0, 'spam': 1}
        self.reverse_label_map = {0: 'ham', 1: 'spam'}
        self.device = device
        self.stemmer = PorterStemmer()
        
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""

        text = str(text)
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text:
            words = text.split()
            words = [self.stemmer.stem(word) for word in words if len(word) > 2]
            return ' '.join(words)
        else:
            return ""
    
    def load_and_preprocess_data(self, filepath):
        print(f"📂 Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        label_col = df.columns[0]
        text_col = df.columns[1]
        
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📊 Columns: {df.columns.tolist()}")
        
        print(f"\n🔍 Checking for NaN values:")
        print(f"NaN in labels: {df[label_col].isna().sum()}")
        print(f"NaN in emails: {df[text_col].isna().sum()}")
        
        initial_count = len(df)
        df = df.dropna(subset=[label_col, text_col])
        dropped_count = initial_count - len(df)
        
        if dropped_count > 0:
            print(f"⚠️ Dropped {dropped_count} rows with NaN values")
        
        print(f"\n📊 Class Distribution:")
        print(df[label_col].value_counts())
        df[label_col] = df[label_col].astype(str).str.lower().str.strip()
        
        valid_labels = ['ham', 'spam']
        df = df[df[label_col].isin(valid_labels)]
        
        print(f"After filtering valid labels: {len(df)} samples")
        print(f"Ham percentage: {(df[label_col] == 'ham').mean() * 100:.2f}%")
        print(f"Spam percentage: {(df[label_col] == 'spam').mean() * 100:.2f}%")
        
        print("🔄 Preprocessing emails...")
        texts = df[text_col].apply(self.preprocess_text)
        empty_mask = texts.str.len() == 0
        if empty_mask.sum() > 0:
            print(f"⚠️ Removing {empty_mask.sum()} emails that became empty after preprocessing")
            texts = texts[~empty_mask]
            df = df[~empty_mask]
        
        labels = df[label_col].map(self.label_map).values
        
        print(f"\n✅ Final dataset: {len(texts)} samples")
        
        return texts, labels
    
    def train(self, filepath, test_size=0.2, epochs=50, batch_size=32, lr=0.001):
        texts, labels = self.load_and_preprocess_data(filepath)
        
        if len(texts) < 10:
            raise ValueError(f"Not enough data after cleaning. Only {len(texts)} samples remaining.")
        
        print("🔄 Creating TF-IDF features...")
        X = self.vectorizer.fit_transform(texts).toarray()
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"📊 Number of features: {X.shape[1]}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"\n📊 Train set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")

        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.model = SpamClassifierNN(X_train.shape[1]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts + 1e-8) 
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
        
        train_losses = []
        test_losses = []
        test_accuracies = []
        best_test_accuracy = 0
        best_model_state = None
        
        print("\n🎯 Starting training...")
        print("="*60)
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion_weighted(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_loss = criterion(test_outputs, y_test).item()
                test_losses.append(test_loss)
                _, predicted = torch.max(test_outputs, 1)
                test_accuracy = (predicted == y_test).sum().item() / len(y_test)
                test_accuracies.append(test_accuracy)
            
            scheduler.step(test_loss)
            
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_model_state = self.model.state_dict().copy()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            
            y_test_np = y_test.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            
            accuracy = accuracy_score(y_test_np, predicted_np)
            precision = precision_score(y_test_np, predicted_np, zero_division=0)
            recall = recall_score(y_test_np, predicted_np, zero_division=0)
            f1 = f1_score(y_test_np, predicted_np, zero_division=0)
            conf_matrix = confusion_matrix(y_test_np, predicted_np)
            
            print("\n" + "="*60)
            print("FINAL MODEL PERFORMANCE")
            print("="*60)
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print("\nConfusion Matrix:")
            print(f"              Predicted")
            print(f"              Ham  Spam")
            if conf_matrix.shape == (2, 2):
                print(f"Actual Ham    {conf_matrix[0][0]:5d}  {conf_matrix[0][1]:5d}")
                print(f"       Spam   {conf_matrix[1][0]:5d}  {conf_matrix[1][1]:5d}")
            else:
                print(f"Confusion matrix shape: {conf_matrix.shape}")
        
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }
    
    def predict(self, text):
        """
        Predict whether an email is spam or ham
        
        Args:
            text: Email text to classify
        
        Returns:
            Dictionary with prediction and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained yet! Please train the model first.")
        
        self.model.eval()
        
        cleaned_text = self.preprocess_text(text)
        
        if not cleaned_text:
            return {
                'prediction': 'ham',  
                'confidence': 0.5,
                'probabilities': {
                    'ham': 0.5,
                    'spam': 0.5}}
        
        X = self.vectorizer.transform([cleaned_text]).toarray()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        prediction = self.reverse_label_map[predicted.item()]
        confidence_score = confidence.item()
        
        return {
            'prediction': prediction,
            'confidence': confidence_score,
            'probabilities': {
                'ham': probabilities[0][0].item(),
                'spam': probabilities[0][1].item()
            }
        }
    
    def predict_batch(self, texts):
        """
        Predict multiple emails
        
        Args:
            texts: List of email texts
        
        Returns:
            List of predictions
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
    
    def save_model(self, path='spam_model'):
        """
        Save trained model to disk
        
        Args:
            path: Base path for saving files
        """
        if self.model is not None:
            torch.save(self.model.state_dict(), f'{path}_weights.pth')
            joblib.dump({
                'vectorizer': self.vectorizer,
                'label_map': self.label_map,
                'reverse_label_map': self.reverse_label_map
            }, f'{path}_components.pkl')
            
            print(f"✅ Model saved to {path}_weights.pth and {path}_components.pkl")
    
    def load_model(self, path='spam_model'):
        """
        Load trained model from disk
        
        Args:
            path: Base path for loading files
        """
        components = joblib.load(f'{path}_components.pkl')
        self.vectorizer = components['vectorizer']
        self.label_map = components['label_map']
        self.reverse_label_map = components['reverse_label_map']
        
        self.model = SpamClassifierNN(self.vectorizer.max_features).to(self.device)
        self.model.load_state_dict(torch.load(f'{path}_weights.pth', 
                                              map_location=self.device))
        self.model.eval()
        
        print(f"✅ Model loaded successfully")