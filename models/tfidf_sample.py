import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

class SpamEmailAgent:
  def __init__(self):
    # Initialize the vectorizer and classifier
    self.vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    self.classifier = MultinomialNB()
    self.is_trained = False

  def preprocess_text(self, text):
    """Basic text preprocessing"""
    if not isinstance(text, str):
      return ""
    # Convert to lowercase and return
    return text.lower().strip()

  def train(self, emails, labels):
    """
    Train the model with email data
    emails: list of email texts
    labels: list of corresponding labels (0 for ham, 1 for spam)
    """
    # Preprocess emails
    processed_emails = [self.preprocess_text(email) for email in emails]

    # Convert text to numerical features
    X = self.vectorizer.fit_transform(processed_emails)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
    )

    # Train the classifier
    self.classifier.fit(X_train, y_train)
    self.is_trained = True

    # Evaluate the model
    y_pred = self.classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

  def predict(self, email):
    """Predict if an email is spam or not"""
    if not self.is_trained:
      raise Exception("Model needs to be trained first!")

    # Preprocess the email
    processed_email = self.preprocess_text(email)

    # Transform using the same vectorizer
    X = self.vectorizer.transform([processed_email])

    # Make prediction
    prediction = self.classifier.predict(X)[0]
    probability = self.classifier.predict_proba(X)[0]

    return {
    'is_spam': bool(prediction),
    'confidence': max(probability) * 100,
    'label': 'spam' if prediction else 'ham'
    }

# Example usage
if __name__ == "__main__":
  # Sample training data (in real case, use a proper dataset)
  sample_emails = [
  "Win a free iPhone now! Click here!",
  "Meeting tomorrow at 10am regarding project",
  "Congratulations! You've won $1000000!",
  "Please review the attached document",
  "URGENT: Your account has been compromised!"
  ]

  sample_labels = [1, 0, 1, 0, 1] # 1 = spam, 0 = ham

  # Create and train the agent
  agent = SpamEmailAgent()
  agent.train(sample_emails, sample_labels)

  # Test with a new email
  test_email = "Claim your prize now before it's too late!"
  result = agent.predict(test_email)

  print("\nPrediction Result:")
  print(f"Email: {test_email}")
  print(f"Is Spam: {result['is_spam']}")
  print(f"Confidence: {result['confidence']:.2f}%")
  print(f"Label: {result['label']}")