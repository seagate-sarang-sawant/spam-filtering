import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
# Assuming preprocess.py has been implemented with necessary functions
from preprocess import (remove_html_tags, remove_urls, remove_punctuation,
                        remove_special_characters, remove_numeric, 
                        remove_non_alphanumeric, replace_chat_words, remove_stopwords,
                        remove_emojis, porter_stemmer)

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    data['cleaned_message'] = data['email'].apply(remove_html_tags)
    data['cleaned_message'] = data['cleaned_message'].apply(remove_urls)
    data['cleaned_message'] = data['cleaned_message'].apply(remove_punctuation)
    data['cleaned_message'] = data['cleaned_message'].apply(remove_special_characters)
    data['cleaned_message'] = data['cleaned_message'].apply(remove_numeric)
    data['cleaned_message'] = data['cleaned_message'].apply(remove_non_alphanumeric)
    data['cleaned_message'] = data['cleaned_message'].apply(replace_chat_words)
    data['cleaned_message'] = data['cleaned_message'].apply(remove_stopwords)
    data['cleaned_message'] = data['cleaned_message'].apply(remove_emojis)
    data['cleaned_message'] = data['cleaned_message'].apply(lambda x: ' '.join([porter_stemmer().stem(word) for word in x.split()]))
    
    return data

def train_model(data):
    X = data['cleaned_message']
    y = data['label']  # Assuming 'label' column contains 'spam' or 'ham'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(CountVectorizer(), LogisticRegression())
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model accuracy: {accuracy:.2f}')
    
    return model

def save_model(model, filename):
    joblib.dump(model, filename)

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    data = load_data(os.path.join(root, 'data/spam.csv'))
    preprocessed_data = preprocess_data(data)
    trained_model = train_model(preprocessed_data)
    save_model(trained_model, 'spam_classifier_model.joblib')