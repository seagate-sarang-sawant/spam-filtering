import os
import pandas as pd
import joblib
from preprocess import preprocess_email

def load_data(file_path):
    return pd.read_csv(file_path)

def load_model(model_path):
    return joblib.load(model_path)

def predict_spam(model, vectorizer, emails):
    processed_emails = [preprocess_email(email) for email in emails]
    email_vectors = vectorizer.transform(processed_emails).toarray()
    predictions = model.predict(email_vectors)
    return predictions

def merge_predictions(original_data, predictions):
    original_data['predicted_label'] = predictions
    return original_data

def save_merged_data(data, file_path):
    data.to_csv(file_path, index=False)

def create_and_save_models():
    # Load the trained model
    model = joblib.load('path/to/your/trained_model.pkl')

    # Use the model for predictions
    predictions = model.predict(new_data)

if __name__ == "__main__":
    root = os.getcwd()
    spam_data_path = '../data/spam.csv'
    model_path = 'path/to/your/trained_model.pkl'  # Update with the actual model path
    vectorizer_path = 'path/to/your/vectorizer.pkl'  # Update with the actual vectorizer path
    pipeline_path = '../spam_classifier_model.joblib'  # Update with the actual pipeline path
    
    # Load data
    data = load_data(spam_data_path)

    # Load model and vectorizer
    #model = load_model(model_path)
    #vectorizer = load_model(vectorizer_path)

    
    # Predict spam
    predictions = predict_spam(model, vectorizer, data['email_text'].tolist())

    # Merge predictions back into the original data
    merged_data = merge_predictions(data, predictions)

    # Save the merged data back to spam.csv
    save_merged_data(merged_data, spam_data_path)