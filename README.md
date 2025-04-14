# spam-filtering
Spam filtering agent 

# Dataset description:
The dataset employed in this project is sourced from a public repository containing 5,572 labeled email or short message service (SMS) entries (Almeida et al., 2011; “spam.csv”). Each entry is categorized as either “spam” or “ham” based on manual or semi-automated labeling. Approximately 13.4% of the data is spam, indicating a moderate class imbalance. Duplicate records are removed to ensure the uniqueness of training examples, and the final dataset is split into training (80%) and testing (20%) sets.

 

# Introduction to Spam Filetring

## 📧 Email as a Primary Communication Tool

    • Email is a cornerstone of digital communication across various sectors

    • Enables fast, low-cost, and global communication

## 🚫 Challenges of Spam Emails

•  Over 50% of email traffic is spam

•  Wastes time, consumes resources, and can carry malware or phishing attempts

•  Overwhelms users and degrade productivity

## ✅ Need for Effective Spam Filtering

•  Enhances user experience by reducing inbox clutter

•  Protects users from scams and security threats

•  Improves productivity and system efficiency

•  Advancements in Machine Learning have enhanced detection capabilities

 

# Problem statement:

 

## 📈 Massive Email Volume

•  Billions of emails sent daily; majority classified as spam

## ⚠️ Emerging Threats

• Spam now includes phishing, malware, scams, and ransomware

• Dynamic techniques make detection harder

## 🧠 Classification Challenge

• Requires accurate distinction between spam and legitimate emails

• Misclassifications can result in loss of critical info or user trust

## 💻 Computational Demand

•  Need for real-time filtering with minimal resource usage

 

 

# Project Objectives

##🤖 Implement Multiple Spam Classifiers

• Develop traditional ML models (e.g., Naive Bayes, Logistic Regression, SVM)

• Integrate advanced models like XGBoost, Random Forest, and LSTM

## 🧪 Benchmark Model Performance

• Evaluate using metrics such as Accuracy, Precision, Recall, and F1-score

• Compare performance on both CountVectorizer and TF-IDF features

## 📊 Analyze Confusion Matrices

• Understand true/false positives and negatives for each model

## 🧠 Leverage Deep Learning (LSTM)

• Assess benefits of using word embeddings and sequence-based learning

## ⚙️ Identify Best-Performing Approach

•  Recommend the most efficient and accurate model for real-world spam detection

 

# Reference:

[Spam Filtering CV, TFIDF, CNN LSTM GloVe]( https://github.com/seagate-sarang-sawant/spam-filtering/blob/main/models/Email_spam_filtering_agent_cv_tfidf_LSTM_GLOVE_Final.ipynb
