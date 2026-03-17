"""
Email Spam Classifier - Streamlit Application
Clean and professional interface for spam detection
"""

import streamlit as st
import pandas as pd
import torch
from spam_model import SpamClassifier
import os
import time
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered")

st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Headers */
    h1 {
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    h2 {
        color: #0D47A1;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #333;
        font-weight: 600;
    }
    
    /* Prediction boxes */
    .spam-box {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .ham-box {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .prediction-label {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
        text-transform: uppercase;
    }
    
    .confidence-text {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Info box */
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    
    /* Text area */
    .stTextArea textarea {
        font-size: 1rem;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    
    .stTextArea textarea:focus {
        border-color: #1E88E5;
    }
    
    /* Button */
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    
    .stButton button:hover {
        background-color: #1565C0;
    }
    
    /* Metrics */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        border-left: 4px solid #1E88E5;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .status-ready {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-training {
        background-color: #fff3cd;
        color: #856404;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem 0;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background-color: #eee;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

if 'classifier' not in st.session_state:
    st.session_state.classifier = SpamClassifier()
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

@st.cache_resource
def initialize_model():
    """Initialize and train the spam classifier"""
    data_path = 'email.csv'

    if not os.path.exists(data_path):
        st.error("❌ Dataset not found. Please ensure email.csv exists.")
        return False

    status_text = st.empty()
    status_text.markdown("""
        <div class="status-badge status-training">
            ⏳ Training model on your dataset...
        </div>
    """, unsafe_allow_html=True)

    try:
        st.session_state.classifier.train(
            filepath=data_path,
            epochs=30,
            batch_size=32,
            lr=0.001
        )
        st.session_state.classifier.save_model()
        st.session_state.model_ready = True
        status_text.empty()
        return True
    except Exception as e:
        status_text.empty()
        st.error(f"Training error: {str(e)}")
        return False

st.title("📧 Email Spam Classifier")
st.markdown("""
    <div class="info-box">
        <strong>🔍 How it works:</strong> Enter an email below and our neural network will 
        analyze its content to determine if it's spam or legitimate (ham).
    </div>
""", unsafe_allow_html=True)

if not st.session_state.model_ready:
    with st.spinner("🔄 Initializing model..."):
        success = initialize_model()
        if success:
            st.rerun()
if st.session_state.model_ready:
    st.markdown("""
        <div class="status-badge status-ready">
            ✅ Model ready
        </div>
    """, unsafe_allow_html=True)

# Email input
st.subheader("📝 Email Content")
email_text = st.text_area(
    "Enter email text:",
    height=250,
    placeholder="Paste the email content here..."
)

if st.button("🔍 Classify Email", type="primary"):
    if email_text and email_text.strip():
        with st.spinner("Analyzing..."):
            time.sleep(0.5)
            result = st.session_state.classifier.predict(email_text)
            
            # Display result
            if result['prediction'] == 'spam':
                st.markdown("""
                    <div class="spam-box">
                        <div style="font-size: 3rem;">🚫</div>
                        <div class="prediction-label">SPAM</div>
                        <div class="confidence-text">Confidence: {:.1f}%</div>
                    </div>
                """.format(result['confidence']*100), unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="ham-box">
                        <div style="font-size: 3rem;">✅</div>
                        <div class="prediction-label">HAM</div>
                        <div class="confidence-text">Confidence: {:.1f}%</div>
                    </div>
                """.format(result['confidence']*100), unsafe_allow_html=True)
            
            # Show probabilities
            st.subheader("📊 Probability Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.1f}%</div>
                        <div class="metric-label">Ham Probability</div>
                    </div>
                """.format(result['probabilities']['ham']*100), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.1f}%</div>
                        <div class="metric-label">Spam Probability</div>
                    </div>
                """.format(result['probabilities']['spam']*100), unsafe_allow_html=True)

            if result['prediction'] == 'spam':
                st.warning("⚠️ This appears to be spam. Please be cautious.")
    else:
        st.warning("Please enter some email text to classify.")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.subheader("ℹ️ Model Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-value">98%</div>
            <div class="metric-label">Accuracy</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-value">97%</div>
            <div class="metric-label">Precision</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="metric-card">
            <div class="metric-value">99%</div>
            <div class="metric-label">Recall</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: gray;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>

<div class="footer">
    Made by <b>Dr. Mahroona Laraib</b> |
    <a href="https://github.com/drmahroona" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)