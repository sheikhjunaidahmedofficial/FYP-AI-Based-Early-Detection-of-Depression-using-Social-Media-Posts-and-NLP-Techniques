# AI-Based Depression Detection Using NLP

### Final Year Project (FYP)

<p align="justify">AI-Based Depression Detection Using NLP is a web-based system that analyzes text input to identify depressive tendencies and emotional states using Natural Language Processing (NLP) and Machine Learning. The application supports dataset-based model training, real-time text analysis, and report management through a professional interface.</p>

<p align="justify">This project is developed as a Final Year Project (FYP) for the BSCS program, focusing on practical AI implementation and mental health awareness.</p>

#

### Project Overview

<p align="justify">Depression is a major mental health concern, and early detection is critical. This system analyzes user-generated text and predicts depressive patterns using a trained machine learning model.</p>

Key functionalities include:

- Uploading labeled datasets for model training
- Real-time analysis of user-written text
- Highlighting depression-related words and phrases
- Maintaining a history of analyzed reports
- Presenting results in a clean, dark-themed interface

**Dataset:** Approximately 20,000 text entries with labels for depression classification.

#

### Objectives

- Detect depressive patterns from textual input
- Apply NLP techniques for text preprocessing and normalization
- Train a machine learning model using labeled datasets
- Predict depression probability for new text
- Highlight depression-related words and phrases
- Store and manage analysis reports
- Provide a user-friendly interface with professional dark theme
- Deploy the system using Streamlit

#

### Technologies and Libraries

**Programming Language**

- Python

**Framework**

- Streamlit

**Machine Learning & NLP**

- Logistic Regression
- TF-IDF Vectorization (unigrams and bigrams)
- Text Cleaning and Normalization
- Keyword and Phrase Extraction

**Libraries Used**

- streamlit
- pandas
- joblib
- time
- json
- os
- re
- base64
- sklearn (TfidfVectorizer, LogisticRegression, train_test_split, accuracy_score)
- emoji

#

### System Features

**Dataset Upload and Training**

- Supports CSV and XLSX files with `post_text` and `label` columns
- Automatic text cleaning and preprocessing
- TF-IDF feature extraction
- Logistic Regression model training with accuracy evaluation
- Model and vectorizer saved locally using joblib

**Text Analysis**

- Real-time depression prediction for user input
- Calculation of depression probability
- Mood classification: Positive, Neutral, Depressed
- Extraction of depression-related words and phrases

**Report Management**

- Stores analyzed reports locally in JSON format
- Displays report history with expandable details
- Highlights depression-related words in the text
- Allows deletion of individual reports

**User Interface**

- Dark-themed professional UI
- Background image with glassmorphism effects
- Responsive wide layout
- Dataset preview and full dataset view
- Progress indicators for depression probability

#

### Text Preprocessing

Text is cleaned and normalized using `utils.py` with the following steps:

- Conversion to lowercase
- Emoji conversion to textual form
- Removal of URLs and mentions
- Retention of hashtag text
- Removal of unnecessary symbols
- Whitespace normalization

#

### Dataset Requirements

- Column `post_text` containing textual data
- Column `label` for depression classification
- Dataset contains approximately 20,000 entries
- Text is automatically cleaned and preprocessed before training or analysis.

#

### Deployment

The application is deployed on Streamlit Cloud and can be accessed online:  
[AI-Based Early Detection of Depression using Social Media Posts and NLP Techniques](https://ai-based-depression-detection-by-using-nlp.streamlit.app)

#

### Disclaimer

This project is developed strictly for academic and research purposes.  
It does not provide medical diagnosis or treatment.  
Users with serious mental health concerns should seek help from qualified professionals.

#

### Team and Supervision

**Team Members:**

- Shaikh Junaid Ahmed
- Muhammad Waleed Irfan
- Muhammad Umair Ansari

**Supervisor:**

- Dr. Sarmad Sheikh

**University:**

- Sindh Madressatul Islam University

#

### License

This project is licensed for academic use only. Commercial use is prohibited.
