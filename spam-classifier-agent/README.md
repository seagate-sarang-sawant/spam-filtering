# Spam Classifier Project

This project implements an AI agent for spam classification using machine learning techniques. The agent is trained on a dataset of emails labeled as either spam or ham, and it can predict the classification of new emails. The results are merged back into the original dataset for further analysis.

## Project Structure

```
spam-classifier
├── data
│   └── spam.csv          # Training data for spam classification
├── src
│   ├── ai_agent.py       # AI agent class for training and prediction
│   ├── preprocess.py      # Functions for preprocessing email text data
│   ├── train_model.py     # Logic for training the classification model
│   └── predict_and_merge.py # Handles prediction and merging results
├── requirements.txt       # Project dependencies
├── .gitignore             # Files and directories to ignore by Git
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites

Make sure you have Python installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd spam-classifier
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Prepare your dataset:
   Ensure that the `data/spam.csv` file is available and contains the labeled email data.

2. Preprocess the data:
   Run the preprocessing script to clean and prepare the data for training:
   ```
   python src/preprocess.py
   ```

3. Train the model:
   Use the training script to train the classification model:
   ```
   python src/train_model.py
   ```

4. Predict and merge results:
   After training, use the prediction script to classify new emails and merge the results back into `spam.csv`:
   ```
   python src/predict_and_merge.py
   ```

### Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

### License

This project is licensed under the MIT License. See the LICENSE file for details.