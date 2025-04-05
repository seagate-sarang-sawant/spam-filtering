class AI_Agent:
    def __init__(self, model=None, vectorizer=None, label_encoder=None):
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder

    def train(self, data_path):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        # Load the data
        data = pd.read_csv(data_path)
        X = data['email']  # Assuming the email text is in a column named 'email'
        y = data['label']  # Assuming the labels are in a column named 'label'

        # Preprocess the data
        from preprocess import preprocess_text
        X_cleaned = X.apply(preprocess_text)

        # Encode the labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_encoded, test_size=0.2, random_state=42)

        # Vectorize the text
        self.vectorizer = CountVectorizer()
        X_train_vectorized = self.vectorizer.fit_transform(X_train)

        # Train the model
        self.model = LogisticRegression()
        self.model.fit(X_train_vectorized, y_train)

    def predict(self, new_emails):
        new_emails_cleaned = [preprocess_text(email) for email in new_emails]
        new_emails_vectorized = self.vectorizer.transform(new_emails_cleaned)
        predictions = self.model.predict(new_emails_vectorized)
        return self.label_encoder.inverse_transform(predictions)

    def merge_predictions(self, data_path, new_emails, predictions):
        import pandas as pd

        # Load the original data
        data = pd.read_csv(data_path)

        # Create a DataFrame for new emails and predictions
        predictions_df = pd.DataFrame({'email': new_emails, 'predicted_label': predictions})

        # Merge with the original data
        merged_data = pd.concat([data, predictions_df], ignore_index=True)

        # Save the merged data back to the original CSV
        merged_data.to_csv(data_path, index=False)