# Spam-Detection-Model
This project aims to develop a spam detection model using machine learning techniques. The model is trained to classify messages as either spam or not spam (ham).
## Overview

Spam detection is a classic problem in natural language processing (NLP) and machine learning. The goal of this project is to build an accurate spam detection model that can effectively classify messages in real-time.
## Dataset

The dataset used for training and evaluation is stored in `spam.csv`. 
## Model

The spam detection model is built using the following steps:
1. Exploratory Data Analysis (EDA) to understand the distribution of spam and ham messages, identify patterns, and preprocess the data.
2. Feature engineering to convert text data into numerical features suitable for machine learning algorithms.
3. Model training using a Naive Bayes classifier, specifically Multinomial Naive Bayes, due to its effectiveness in text classification tasks.
4. Model evaluation using various metrics such as accuracy, precision, recall, and F1-score.

## Technologies Used

- Python: Programming language used for data preprocessing, model training, and development of the user interface.
- Pandas: Data manipulation library used for handling the dataset.
- Scikit-learn: Machine learning library used for building and evaluating the spam detection model.
- Streamlit: Python library used for creating a user-friendly web application for interacting with the spam detection model.

## Usage

To run the spam detection app locally, follow these steps:
1. Install the required Python packages listed in `requirements.txt`.
2. Run the Streamlit app using the command `streamlit run app.py`.
3. Input a message in the provided text box and click the "Predict" button to see the model's classification result.

## Future Work

Possible future enhancements to the project include:
- Experimenting with different machine learning algorithms and fine-tuning hyperparameters to improve model performance.
