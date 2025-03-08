# Spam Email Classifier

## Overview
This project is a **Spam Email Classifier** built using **TensorFlow, Keras, and Streamlit**. It utilizes an **LSTM-based neural network** to classify emails as spam or ham. The model is trained on a labeled dataset of **5,572 emails** and deployed with a user-friendly **Streamlit** interface for real-time predictions. The app also provides **training accuracy and loss curves** for model evaluation.

## Features
- **Deep Learning Model**: Uses an LSTM-based neural network for text classification.
- **Text Preprocessing**: Includes tokenization, stopword removal, stemming, and padding.
- **Interactive UI**: Built with Streamlit for easy email classification.
- **Model Training**: Implemented using TensorFlow and Keras.
- **Visualization**: Displays training accuracy and loss curves.

## Technologies Used
- Python
- TensorFlow & Keras
- Scikit-learn
- Natural Language Processing (NLP)
- Streamlit
- Pandas & NumPy
- Matplotlib
- Joblib (for model serialization)

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8). Install dependencies using:
```bash
pip install tensorflow keras numpy pandas scikit-learn streamlit joblib matplotlib nltk
```

## Model Training
Run the following script to train the model:
```python
python train_model.py
```
This script:
1. Loads and preprocesses the dataset.
2. Tokenizes and pads email text.
3. Trains an LSTM model.
4. Saves the trained model (`my_model.keras`) and tokenizer (`tokenizer.pkl`).
5. Stores training history (`training_history.pkl`).

## Running the Streamlit App
Once the model is trained, run the Streamlit app using:
```bash
streamlit run app.py
```
The UI will allow users to input email text and classify it as spam or ham. Users can also **view training accuracy and loss graphs** within the app.

## Deployment
To deploy the Streamlit app:
1. Use **Streamlit Sharing** or **Heroku**.
2. Upload the model files (`my_model.keras`, `tokenizer.pkl`) to the server.
3. Run `streamlit run app.py` in the deployment environment.

## Usage
1. Enter an email text in the input box.
2. Click the **Predict** button.
3. The app displays whether the email is spam or ham.
4. Optionally, view the **model training curves**.

## Future Improvements
- Implement **attention mechanisms** for better text classification.
- Expand the dataset for improved accuracy.
- Integrate **Flask/FastAPI** for a REST API.
- Deploy on a cloud platform for scalability.


