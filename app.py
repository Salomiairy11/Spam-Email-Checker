import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_model.keras')

@st.cache_resource
def load_tokenizer():
    return joblib.load('tokenizer.pkl')

model = load_model()
tokenizer = load_tokenizer()

st.title('Spam Email Classifier')
st.markdown("Enter the email content below and click **Predict** to check if it's Spam or Ham.")

email_input = st.text_area("Email Content")

if st.button("Predict"):
    if email_input.strip():
        try:
            sequences = tokenizer.texts_to_sequences([email_input])
            
            if not sequences or not sequences[0]:
                st.error("⚠️ Tokenization resulted in an empty sequence. Try different input.")
            else:
                padded_sequence = pad_sequences(sequences, maxlen=100)

                prediction = model.predict(padded_sequence)[0][0]

                if prediction >= 0.5:
                    st.success("✅ This email is Not Spam.")
                else:
                    st.warning("🚨 This email is **SPAM**.")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.error("⚠️ Please enter an email to classify.")


def plot_training_curves():
    try:
        history = joblib.load('training_history.pkl')  
        
        if not isinstance(history, dict) or 'accuracy' not in history:
            st.error("Invalid training history file. Please check 'training_history.pkl'.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(history['accuracy'], label='Training Accuracy')
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()

        axes[1].plot(history['loss'], label='Training Loss')
        axes[1].plot(history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].legend()

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading training history: {e}")

if st.button('Show Training Curves'):
    plot_training_curves()
